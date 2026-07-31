"""Avalia alerta térmico personalizado por ovelha e relativo ao rebanho.

O alvo exploratório é monta no dia da fotografia ou no dia seguinte. Como a
base não contém horários, ele representa dias de calendário, não 24 horas
cronometradas.

Os atributos personalizados usam somente medições anteriores do mesmo animal.
Os atributos relativos ao rebanho usam medições sem rótulo do mesmo dia e a
mediana deixa a própria ovelha de fora. Esse segundo conjunto pressupõe o
cenário operacional em que o lote é fotografado antes da emissão dos alertas.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter
import warnings

import numpy as np
import pandas as pd

from avaliar_janela_24h import (
    DEFAULT_DATA_DIR,
    DEFAULT_INPUT,
    GROUP_COLUMNS,
    aggregate_metrics,
    append_progress,
    calculate_metrics,
    make_grouped_splits,
    prediction_scores,
    prepare_dataset,
    purge_adjacent_dates,
    select_f1_threshold,
)
from bootstrap_janela_24h import cluster_bootstrap
from extrair_roi import PROJECT_ROOT
from modelar_estro import parse_locale_numeric


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modeling_personalized_alert"
DEFAULT_CANDIDATE_CACHE = (
    PROJECT_ROOT
    / "outputs"
    / "modeling_window_24h"
    / "candidate_features.csv"
)
TARGET_NAME = "mount_today_or_next_day"
TARGET_COLUMN = "target_mount_today_or_next_day"
MODEL_NAMES = ("logistic", "svm_rbf", "hist_gradient_boosting")
BASE_COLUMNS = (
    "fixed_15_temp_mean",
    "fixed_15_temp_median",
    "fixed_15_temp_p90",
    "fixed_15_temp_max",
    "fixed_15_temp_std",
    "ambient_temperature_c",
)
HERD_SOURCES = (
    "fixed_15_temp_mean",
    "fixed_15_temp_p90",
    "fixed_15_temp_max",
    "poi_temperature",
)
PERSONAL_SOURCES = (*HERD_SOURCES, "ambient_temperature_c")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object


def leave_one_out_group_median(
    values: pd.Series,
    groups: pd.Series,
) -> pd.Series:
    """Calcula a mediana do grupo sem usar o valor da própria linha."""

    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=values.index, dtype=float)
    frame = pd.DataFrame({"value": numeric, "group": groups})
    for _, group in frame.groupby("group", sort=False, dropna=False):
        group_values = group["value"].to_numpy(dtype=float)
        group_indices = group.index.to_numpy()
        for position, index in enumerate(group_indices):
            other = np.delete(group_values, position)
            finite = other[np.isfinite(other)]
            if finite.size:
                result.loc[index] = float(np.median(finite))
    return result


def add_personalized_features(
    dataframe: pd.DataFrame,
    *,
    history_window: int = 5,
) -> pd.DataFrame:
    """Cria atributos contemporâneos do lote e históricos causais."""

    if history_window < 1:
        raise ValueError("A janela de histórico deve ser positiva.")
    required = {
        "animal_group",
        "collection_datetime",
        "collection_date",
        "source_csv_row",
        *PERSONAL_SOURCES,
    }
    missing = required - set(dataframe)
    if missing:
        raise ValueError(
            "Colunas ausentes para personalização: "
            + ", ".join(sorted(missing))
        )
    result = dataframe.copy()
    for source in HERD_SOURCES:
        herd_median = leave_one_out_group_median(
            result[source],
            result["collection_date"],
        )
        result[f"{source}_minus_herd_median"] = result[source] - herd_median
        result[f"{source}_herd_percentile"] = result.groupby(
            "collection_date",
            sort=False,
        )[source].rank(method="average", pct=True)

    temporal = result.copy()
    temporal["_original_order"] = np.arange(len(temporal))
    temporal = temporal.sort_values(
        ["animal_group", "collection_datetime", "source_csv_row"],
        kind="stable",
    )
    previous_date = temporal.groupby(
        "animal_group",
        sort=False,
    )["collection_datetime"].shift(1)
    temporal["days_since_previous"] = (
        temporal["collection_datetime"] - previous_date
    ).dt.days.astype(float)
    temporal["has_previous_measurement"] = previous_date.notna().astype("int8")
    positive_interval = temporal["days_since_previous"].where(
        temporal["days_since_previous"] > 0
    )

    for source in PERSONAL_SOURCES:
        previous = temporal.groupby("animal_group", sort=False)[source].shift(1)
        recent_median = previous.groupby(
            temporal["animal_group"],
            sort=False,
        ).transform(
            lambda series: series.rolling(
                history_window,
                min_periods=1,
            ).median()
        )
        delta = temporal[source] - previous
        temporal[f"{source}_delta_previous"] = delta
        temporal[f"{source}_delta_per_day"] = delta / positive_interval
        temporal[f"{source}_minus_personal_median"] = (
            temporal[source] - recent_median
        )

    for source in HERD_SOURCES:
        relative = f"{source}_minus_herd_median"
        previous_relative = temporal.groupby(
            "animal_group",
            sort=False,
        )[relative].shift(1)
        recent_relative_median = previous_relative.groupby(
            temporal["animal_group"],
            sort=False,
        ).transform(
            lambda series: series.rolling(
                history_window,
                min_periods=1,
            ).median()
        )
        temporal[f"{relative}_minus_personal_median"] = (
            temporal[relative] - recent_relative_median
        )

    return (
        temporal.sort_values("_original_order", kind="stable")
        .drop(columns="_original_order")
        .reset_index(drop=True)
    )


def feature_sets() -> dict[str, list[str]]:
    herd_columns = [
        derived
        for source in HERD_SOURCES
        for derived in (
            f"{source}_minus_herd_median",
            f"{source}_herd_percentile",
        )
    ]
    personal_columns = [
        derived
        for source in PERSONAL_SOURCES
        for derived in (
            f"{source}_delta_previous",
            f"{source}_delta_per_day",
            f"{source}_minus_personal_median",
        )
    ]
    double_difference_columns = [
        f"{source}_minus_herd_median_minus_personal_median"
        for source in HERD_SOURCES
    ]
    history_columns = [
        "days_since_previous",
        "has_previous_measurement",
    ]
    return {
        "ambient_only": ["ambient_temperature_c"],
        "fixed_15": list(BASE_COLUMNS),
        "herd_relative": herd_columns,
        "personalized": [*personal_columns, *history_columns],
        "combined": [
            *BASE_COLUMNS,
            *herd_columns,
            *personal_columns,
            *double_difference_columns,
            *history_columns,
        ],
        "rectal_control": [
            "rectal_spot_temperature",
            "rectal_mean_temperature",
            "ambient_temperature_c",
        ],
    }


def build_model(name: str, *, seed: int) -> ModelSpec:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if name == "logistic":
        estimator = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                C=0.3,
                class_weight="balanced",
                max_iter=5_000,
                random_state=seed,
            ),
        )
    elif name == "svm_rbf":
        estimator = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SVC(
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                random_state=seed,
            ),
        )
    elif name == "hist_gradient_boosting":
        estimator = make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=60,
                max_leaf_nodes=7,
                min_samples_leaf=20,
                l2_regularization=1.0,
                class_weight="balanced",
                random_state=seed,
            ),
        )
    else:
        raise ValueError(f"Modelo desconhecido: {name}")
    return ModelSpec(name=name, estimator=estimator)


def tune_threshold_inside_training(
    dataframe: pd.DataFrame,
    features: pd.DataFrame,
    target: np.ndarray,
    outer_train_index: np.ndarray,
    *,
    model_name: str,
    grouping: str,
    inner_folds: int,
    seed: int,
    purge_days: int,
) -> tuple[float, float, int]:
    from sklearn.model_selection import StratifiedGroupKFold

    train_frame = dataframe.iloc[outer_train_index].reset_index(drop=True)
    train_features = features.iloc[outer_train_index].reset_index(drop=True)
    train_target = target[outer_train_index]
    groups = train_frame[GROUP_COLUMNS[grouping]].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=inner_folds,
        shuffle=True,
        random_state=seed,
    )
    inner_truth: list[np.ndarray] = []
    inner_scores: list[np.ndarray] = []
    used_folds = 0
    placeholder = np.zeros((len(train_frame), 1))

    for inner_fold, (inner_train, inner_test) in enumerate(
        splitter.split(placeholder, train_target, groups),
        start=1,
    ):
        if grouping == "date":
            inner_train = purge_adjacent_dates(
                inner_train,
                inner_test,
                train_frame["collection_datetime"],
                purge_days=purge_days,
            )
        if np.unique(train_target[inner_train]).size < 2:
            continue
        if np.unique(train_target[inner_test]).size < 2:
            continue
        model = build_model(
            model_name,
            seed=seed + inner_fold,
        ).estimator
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(
                train_features.iloc[inner_train],
                train_target[inner_train],
            )
        inner_truth.append(train_target[inner_test])
        inner_scores.append(
            prediction_scores(model, train_features.iloc[inner_test])
        )
        used_folds += 1

    if not inner_truth:
        default = 0.0 if model_name == "svm_rbf" else 0.5
        return default, np.nan, 0
    return (
        *select_f1_threshold(
            np.concatenate(inner_truth),
            np.concatenate(inner_scores),
        ),
        used_folds,
    )


def event_starts(
    dataframe: pd.DataFrame,
    *,
    merge_positive_gap_days: int,
) -> pd.DataFrame:
    """Retorna inícios de episódios, agrupando positivos próximos."""

    positive = dataframe.loc[
        dataframe["target_mount_today"].eq(1),
        ["animal_group", "collection_datetime"],
    ].sort_values(["animal_group", "collection_datetime"])
    previous_positive = positive.groupby(
        "animal_group",
        sort=False,
    )["collection_datetime"].shift(1)
    starts = previous_positive.isna() | (
        positive["collection_datetime"] - previous_positive
    ).dt.days.gt(merge_positive_gap_days)
    return positive.loc[starts].reset_index(drop=True)


def alert_episode_starts(
    dataframe: pd.DataFrame,
    *,
    cooldown_days: int,
) -> pd.DataFrame:
    alerts = dataframe.loc[
        dataframe["prediction"].eq(1),
        ["animal_group", "collection_datetime"],
    ].sort_values(["animal_group", "collection_datetime"])
    previous_alert = alerts.groupby(
        "animal_group",
        sort=False,
    )["collection_datetime"].shift(1)
    starts = previous_alert.isna() | (
        alerts["collection_datetime"] - previous_alert
    ).dt.days.gt(cooldown_days)
    return alerts.loc[starts].reset_index(drop=True)


def calculate_event_metrics(
    dataframe: pd.DataFrame,
    *,
    merge_positive_gap_days: int,
    alert_window_days: int = 1,
    alert_cooldown_days: int = 1,
) -> dict[str, float | int]:
    """Faz pareamento um-para-um entre alertas e episódios verdadeiros."""

    events = event_starts(
        dataframe,
        merge_positive_gap_days=merge_positive_gap_days,
    )
    alerts = alert_episode_starts(
        dataframe,
        cooldown_days=alert_cooldown_days,
    )
    available_alerts = set(alerts.index.tolist())
    matched_alerts: set[int] = set()
    matched_events = 0

    for event in events.itertuples(index=False):
        candidates = alerts[
            alerts["animal_group"].eq(event.animal_group)
            & alerts["collection_datetime"].between(
                event.collection_datetime
                - pd.Timedelta(days=alert_window_days),
                event.collection_datetime,
            )
        ]
        candidate_indices = [
            index
            for index in candidates.index
            if index in available_alerts
        ]
        if not candidate_indices:
            continue
        chosen = max(
            candidate_indices,
            key=lambda index: alerts.loc[index, "collection_datetime"],
        )
        available_alerts.remove(chosen)
        matched_alerts.add(chosen)
        matched_events += 1

    event_count = len(events)
    alert_count = len(alerts)
    false_alerts = alert_count - len(matched_alerts)
    return {
        "event_count": event_count,
        "detected_events": matched_events,
        "event_recall": (
            matched_events / event_count if event_count else np.nan
        ),
        "alert_episodes": alert_count,
        "matched_alert_episodes": len(matched_alerts),
        "alert_precision": (
            len(matched_alerts) / alert_count if alert_count else np.nan
        ),
        "false_alert_episodes": false_alerts,
        "false_alerts_per_100_animal_days": (
            100 * false_alerts / len(dataframe) if len(dataframe) else np.nan
        ),
    }


def event_metric_tables(
    predictions: pd.DataFrame,
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = dataframe[
        [
            "source_csv_row",
            "collection_datetime",
            "target_mount_today",
        ]
    ]
    rows: list[dict[str, object]] = []
    group_columns = ["grouping", "feature_set", "model", "repeat"]
    for keys, subset in predictions.groupby(group_columns, sort=False):
        stream = subset.merge(
            metadata,
            on="source_csv_row",
            how="left",
            validate="one_to_one",
        )
        for merge_gap in (1, 2):
            rows.append(
                {
                    **dict(zip(group_columns, keys, strict=True)),
                    "merge_positive_gap_days": merge_gap,
                    **calculate_event_metrics(
                        stream,
                        merge_positive_gap_days=merge_gap,
                    ),
                }
            )
    by_repeat = pd.DataFrame(rows)
    metric_columns = [
        "event_count",
        "detected_events",
        "event_recall",
        "alert_episodes",
        "matched_alert_episodes",
        "alert_precision",
        "false_alert_episodes",
        "false_alerts_per_100_animal_days",
    ]
    summary = by_repeat.groupby(
        [
            "grouping",
            "feature_set",
            "model",
            "merge_positive_gap_days",
        ],
        as_index=False,
    )[metric_columns].agg(["mean", "std", "count"])
    summary.columns = [
        "_".join(item for item in column if item)
        for column in summary.columns.to_flat_index()
    ]
    return by_repeat, summary


def daily_ranking_tables(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Avalia a ordenação de ovelhas dentro de cada data de teste."""

    date_predictions = predictions[predictions["grouping"].eq("date")].copy()
    rows: list[dict[str, object]] = []
    config_columns = ["feature_set", "model", "repeat"]
    for keys, subset in date_predictions.groupby(config_columns, sort=False):
        positives = int(subset["truth"].sum())
        daily_ap: list[float] = []
        top_k_totals = {
            top_k: {"selected": 0, "true_positive": 0.0}
            for top_k in (3, 5)
        }
        from sklearn.metrics import average_precision_score

        for _, day in subset.groupby("collection_date", sort=False):
            if day["truth"].sum() > 0:
                daily_ap.append(
                    float(average_precision_score(day["truth"], day["score"]))
                )
            for top_k in top_k_totals:
                true_positive, selected = expected_top_k_counts(
                    day["truth"].to_numpy(dtype=int),
                    day["score"].to_numpy(dtype=float),
                    top_k=top_k,
                )
                top_k_totals[top_k]["true_positive"] += true_positive
                top_k_totals[top_k]["selected"] += selected
        for top_k in (3, 5):
            selected = int(top_k_totals[top_k]["selected"])
            true_selected = float(
                top_k_totals[top_k]["true_positive"]
            )
            rows.append(
                {
                    **dict(zip(config_columns, keys, strict=True)),
                    "top_k": top_k,
                    "selected_records": selected,
                    "positive_records": positives,
                    "true_positives_at_k": true_selected,
                    "precision_at_k": (
                        true_selected / selected if selected else np.nan
                    ),
                    "recall_at_k": (
                        true_selected / positives if positives else np.nan
                    ),
                    "mean_daily_average_precision": (
                        float(np.mean(daily_ap)) if daily_ap else np.nan
                    ),
                    "dates_with_positive": len(daily_ap),
                }
            )
    by_repeat = pd.DataFrame(rows)
    metric_columns = [
        "precision_at_k",
        "recall_at_k",
        "mean_daily_average_precision",
    ]
    summary = by_repeat.groupby(
        ["feature_set", "model", "top_k"],
        as_index=False,
    )[metric_columns].agg(["mean", "std", "count"])
    summary.columns = [
        "_".join(item for item in column if item)
        for column in summary.columns.to_flat_index()
    ]
    return by_repeat, summary


def expected_top_k_counts(
    truth: np.ndarray,
    score: np.ndarray,
    *,
    top_k: int,
) -> tuple[float, int]:
    """Retorna TP esperado no top-k, repartindo empates no ponto de corte."""

    if top_k < 1:
        raise ValueError("top_k deve ser positivo.")
    truth = np.asarray(truth, dtype=int)
    score = np.asarray(score, dtype=float)
    if truth.shape != score.shape:
        raise ValueError("Rótulos e escores devem ter a mesma forma.")
    selected = min(top_k, len(score))
    if selected == 0:
        return 0.0, 0
    cutoff = np.sort(score)[-selected]
    above = score > cutoff
    tied = score == cutoff
    remaining = selected - int(above.sum())
    tied_count = int(tied.sum())
    expected_tied_positive = (
        remaining * float(truth[tied].sum()) / tied_count
        if tied_count
        else 0.0
    )
    return float(truth[above].sum()) + expected_tied_positive, selected


def prepare_personalized_dataset(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    dataframe, _ = prepare_dataset(
        args.input.expanduser().resolve(),
        data_dir=args.data_dir.expanduser().resolve(),
        collection_year=args.collection_year,
        feature_cache=args.candidate_cache.expanduser().resolve(),
    )
    dataframe["rectal_spot_temperature"] = parse_locale_numeric(
        dataframe["Temp. Retal"]
    )
    dataframe["rectal_mean_temperature"] = parse_locale_numeric(
        dataframe["Media T Retal"]
    )
    personalized = add_personalized_features(
        dataframe,
        history_window=args.history_window,
    )
    return personalized, feature_sets()


def audit_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "records", "value": len(dataframe)},
            {
                "metric": "animals",
                "value": dataframe["animal_group"].nunique(),
            },
            {
                "metric": "dates",
                "value": dataframe["collection_date"].nunique(),
            },
            {
                "metric": "mount_today_positives",
                "value": int(dataframe["target_mount_today"].sum()),
            },
            {
                "metric": "alert_target_positives",
                "value": int(dataframe[TARGET_COLUMN].sum()),
            },
            {
                "metric": "events_merge_gap_1",
                "value": len(
                    event_starts(
                        dataframe,
                        merge_positive_gap_days=1,
                    )
                ),
            },
            {
                "metric": "events_merge_gap_2",
                "value": len(
                    event_starts(
                        dataframe,
                        merge_positive_gap_days=2,
                    )
                ),
            },
        ]
    )


def evaluate(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.log"
    progress_path.write_text("", encoding="utf-8")
    dataframe, all_feature_sets = prepare_personalized_dataset(args)
    selected_features = {
        name: all_feature_sets[name] for name in args.feature_sets
    }
    target = dataframe[TARGET_COLUMN].to_numpy(dtype=int)
    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    start_time = perf_counter()
    total = len(args.groupings) * len(selected_features) * len(args.models)
    completed = 0

    for grouping in args.groupings:
        splits = make_grouped_splits(
            dataframe,
            target,
            grouping=grouping,
            folds=args.outer_folds,
            repeats=args.repeats,
            seed=args.seed,
            purge_days=args.purge_days,
        )
        for repeat, fold, train_index, test_index in splits:
            split_rows.append(
                {
                    "target": TARGET_NAME,
                    "grouping": grouping,
                    "repeat": repeat,
                    "fold": fold,
                    "train_records_after_purge": len(train_index),
                    "test_records": len(test_index),
                    "train_positives": int(target[train_index].sum()),
                    "test_positives": int(target[test_index].sum()),
                    "train_groups": dataframe.iloc[train_index][
                        GROUP_COLUMNS[grouping]
                    ].nunique(),
                    "test_groups": dataframe.iloc[test_index][
                        GROUP_COLUMNS[grouping]
                    ].nunique(),
                }
            )
        for feature_name, columns in selected_features.items():
            features = dataframe[columns]
            for model_name in args.models:
                for repeat, fold, train_index, test_index in splits:
                    threshold, inner_f1, inner_used = (
                        tune_threshold_inside_training(
                            dataframe,
                            features,
                            target,
                            train_index,
                            model_name=model_name,
                            grouping=grouping,
                            inner_folds=args.inner_folds,
                            seed=args.seed + repeat * 1_000 + fold * 10,
                            purge_days=args.purge_days,
                        )
                    )
                    model = build_model(
                        model_name,
                        seed=args.seed + repeat * 100 + fold,
                    ).estimator
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(
                            features.iloc[train_index],
                            target[train_index],
                        )
                    test_score = prediction_scores(
                        model,
                        features.iloc[test_index],
                    )
                    test_prediction = (test_score >= threshold).astype(int)
                    metrics = calculate_metrics(
                        target[test_index],
                        test_prediction,
                        test_score,
                    )
                    fold_rows.append(
                        {
                            "target": TARGET_NAME,
                            "grouping": grouping,
                            "repeat": repeat,
                            "fold": fold,
                            "feature_set": feature_name,
                            "model": model_name,
                            "threshold": threshold,
                            "inner_f1_at_threshold": inner_f1,
                            "inner_folds_used": inner_used,
                            "train_records": len(train_index),
                            "test_records": len(test_index),
                            "test_prevalence": float(
                                target[test_index].mean()
                            ),
                            **metrics,
                        }
                    )
                    for index, score, prediction in zip(
                        test_index,
                        test_score,
                        test_prediction,
                        strict=True,
                    ):
                        row = dataframe.iloc[index]
                        prediction_rows.append(
                            {
                                "target": TARGET_NAME,
                                "grouping": grouping,
                                "repeat": repeat,
                                "fold": fold,
                                "feature_set": feature_name,
                                "model": model_name,
                                "source_csv_row": int(row["source_csv_row"]),
                                "animal_group": str(row["animal_group"]),
                                "collection_date": row["collection_date"],
                                "truth": int(target[index]),
                                "score": float(score),
                                "threshold": float(threshold),
                                "prediction": int(prediction),
                            }
                        )
                completed += 1
                append_progress(
                    progress_path,
                    (
                        f"{completed}/{total} {grouping} {feature_name} "
                        f"{model_name} elapsed_s={perf_counter() - start_time:.1f}"
                    ),
                )

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = aggregate_metrics(folds)
    split_audit = pd.DataFrame(split_rows).drop_duplicates()
    event_by_repeat, event_summary = event_metric_tables(
        predictions,
        dataframe,
    )
    daily_by_repeat, daily_summary = daily_ranking_tables(predictions)
    definitions = pd.DataFrame(
        [
            {
                "feature_set": name,
                "columns": "|".join(columns),
                "feature_count": len(columns),
            }
            for name, columns in selected_features.items()
        ]
    )

    outputs = (
        (audit_rows(dataframe), "data_audit.csv"),
        (definitions, "feature_definitions.csv"),
        (split_audit, "split_audit.csv"),
        (folds, "fold_metrics.csv"),
        (predictions, "out_of_fold_predictions.csv"),
        (summary, "summary_metrics.csv"),
        (event_by_repeat, "event_metrics_by_repeat.csv"),
        (event_summary, "event_metrics_summary.csv"),
        (daily_by_repeat, "daily_ranking_by_repeat.csv"),
        (daily_summary, "daily_ranking_summary.csv"),
    )
    for output, filename in outputs:
        output.to_csv(
            output_dir / filename,
            sep=";",
            index=False,
            encoding="utf-8-sig",
        )

    if args.bootstrap_iterations > 0 and "logistic" in args.models:
        averaged = (
            predictions[
                predictions["model"].eq("logistic")
            ]
            .groupby(
                [
                    "grouping",
                    "feature_set",
                    "source_csv_row",
                    "truth",
                    "animal_group",
                    "collection_date",
                ],
                as_index=False,
            )["score"]
            .mean()
            .rename(
                columns={
                    "animal_group": "animal",
                    "collection_date": "date",
                }
            )
        )
        absolute, differences = cluster_bootstrap(
            averaged,
            reference="fixed_15",
            iterations=args.bootstrap_iterations,
            seed=args.seed,
        )
        absolute.to_csv(
            output_dir / "bootstrap_absolute_metrics.csv",
            sep=";",
            index=False,
            encoding="utf-8-sig",
        )
        differences.to_csv(
            output_dir / "bootstrap_feature_differences.csv",
            sep=";",
            index=False,
            encoding="utf-8-sig",
        )

    append_progress(
        progress_path,
        f"completed elapsed_s={perf_counter() - start_time:.1f}",
    )
    print(audit_rows(dataframe).to_string(index=False))
    print(
        summary.sort_values(
            ["grouping", "average_precision_mean"],
            ascending=[True, False],
        )
        .groupby("grouping")
        .head(8)
        .to_string(index=False)
    )
    print(event_summary.to_string(index=False))
    print(daily_summary.to_string(index=False))
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Avalia normalização pelo rebanho, baseline individual e alerta "
            "por evento com validação agrupada."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--candidate-cache",
        type=Path,
        default=DEFAULT_CANDIDATE_CACHE,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--collection-year", type=int, default=2025)
    parser.add_argument("--history-window", type=int, default=5)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--purge-days", type=int, default=1)
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--groupings",
        nargs="+",
        choices=tuple(GROUP_COLUMNS),
        default=list(GROUP_COLUMNS),
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=tuple(feature_sets()),
        default=list(feature_sets()),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(MODEL_NAMES),
    )
    return parser


def main() -> int:
    try:
        return evaluate(make_parser().parse_args())
    except Exception as error:
        print(f"Erro: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
