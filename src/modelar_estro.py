"""Modelagem preliminar do estro com validação estratificada e agrupada.

O conjunto de teste nunca compartilha com o treino a unidade escolhida em
``--group-by``: animal ou data de coleta normalizada.
A análise principal é a combinação ``fixed_11 + logistic_regression``; os
demais métodos são comparações e análises de sensibilidade pré-especificadas.

Política provisória do rótulo:

* ``Monta == true`` -> estro (1);
* célula vazia em ``Monta`` -> não estro (0).

Essa política depende do protocolo de campo e precisa ser confirmada pelos
responsáveis pela coleta antes da análise científica final.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from extrair_roi import PROJECT_ROOT


DEFAULT_INPUT = (
    PROJECT_ROOT
    / "outputs"
    / "roi_multiescala"
    / "roi_features_comparative.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modeling_grouped"
PRIMARY_FEATURE_SET = "fixed_11"
PRIMARY_MODEL = "logistic_regression"
VALIDATION_GROUP_COLUMNS = {
    "animal": "animal_group",
    "date": "collection_date",
}


FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "ambient_only": ("ambient_temperature_c",),
    "field_spot": ("field_spot_temperature", "ambient_temperature_c"),
    "radiometric_poi": ("poi_temperature", "ambient_temperature_c"),
    "fixed_7": (
        "fixed_7_temp_mean",
        "fixed_7_temp_median",
        "fixed_7_temp_p90",
        "fixed_7_temp_max",
        "fixed_7_temp_std",
        "ambient_temperature_c",
    ),
    "fixed_11": (
        "fixed_11_temp_mean",
        "fixed_11_temp_median",
        "fixed_11_temp_p90",
        "fixed_11_temp_max",
        "fixed_11_temp_std",
        "ambient_temperature_c",
    ),
    "fixed_15": (
        "fixed_15_temp_mean",
        "fixed_15_temp_median",
        "fixed_15_temp_p90",
        "fixed_15_temp_max",
        "fixed_15_temp_std",
        "ambient_temperature_c",
    ),
    "region_growing": (
        "rg_temp_mean",
        "rg_temp_median",
        "rg_temp_p90",
        "rg_temp_max",
        "rg_temp_std",
        "ambient_temperature_c",
    ),
}


def parse_locale_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    normalized = (
        series.astype("string")
        .str.strip()
        .str.replace(",", ".", regex=False)
        .replace({"": pd.NA, "nan": pd.NA, "<NA>": pd.NA})
    )
    return pd.to_numeric(normalized, errors="coerce")


def encode_monta(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    positive = {"true", "1", "sim", "yes"}
    negative = {"", "false", "0", "não", "nao", "no"}
    encoded = pd.Series(index=series.index, dtype="int8")

    missing = series.isna() | normalized.isna()
    encoded.loc[missing] = 0
    encoded.loc[normalized.isin(positive)] = 1
    encoded.loc[normalized.isin(negative)] = 0

    unknown = encoded.isna()
    if unknown.any():
        values = sorted(normalized.loc[unknown].dropna().unique().tolist())
        raise ValueError(
            "Valores de Monta não reconhecidos: " + ", ".join(values)
        )
    return encoded.astype("int8")


def normalize_collection_dates(
    series: pd.Series,
    *,
    collection_year: int = 2025,
) -> tuple[pd.Series, pd.Series]:
    """Normaliza o ano da coleta sem alterar a planilha original.

    A planilha contém uma sequência evidente de anos incrementais entre 2026
    e 2033 em fotografias consecutivas de fevereiro. O dia e o mês são
    preservados, e somente o ano é corrigido na cópia analítica.
    """

    original = series.astype("string").str.strip()
    parsed = pd.to_datetime(
        original,
        format="%d.%m.%Y",
        errors="coerce",
    )
    invalid = parsed.isna()
    if invalid.any():
        values = sorted(original.loc[invalid].dropna().unique().tolist())
        raise ValueError(
            "Datas de coleta inválidas: " + ", ".join(values)
        )

    corrected = parsed.dt.year.ne(collection_year)
    normalized = pd.to_datetime(
        parsed.dt.strftime(f"%d.%m.{collection_year}"),
        format="%d.%m.%Y",
        errors="raise",
    )
    return normalized.dt.strftime("%Y-%m-%d"), corrected.astype(bool)


def prepare_dataset(
    dataframe: pd.DataFrame,
    *,
    collection_year: int = 2025,
) -> tuple[pd.DataFrame, dict[str, object]]:
    required = {
        "Data",
        "id",
        "Monta",
        "fixed_15_available",
        "Termograma ",
        "source_csv_row",
    }
    required.update(
        column for columns in FEATURE_SETS.values() for column in columns
        if column != "field_spot_temperature"
    )
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(
            "Colunas necessárias ausentes: " + ", ".join(sorted(missing))
        )

    prepared = dataframe.copy()
    prepared["target_monta"] = encode_monta(prepared["Monta"])
    prepared["animal_group"] = (
        prepared["id"]
        .astype("string")
        .str.replace(r"\.0$", "", regex=True)
    )
    prepared["collection_date_original"] = (
        prepared["Data"].astype("string").str.strip()
    )
    (
        prepared["collection_date"],
        prepared["date_year_corrected"],
    ) = normalize_collection_dates(
        prepared["Data"],
        collection_year=collection_year,
    )
    prepared["field_spot_temperature"] = parse_locale_numeric(
        prepared["Termograma "]
    )

    numeric_columns = sorted(
        {
            column
            for columns in FEATURE_SETS.values()
            for column in columns
        }
    )
    for column in numeric_columns:
        prepared[column] = parse_locale_numeric(prepared[column])

    fixed_15_available = (
        prepared["fixed_15_available"]
        .astype("string")
        .str.lower()
        .map({"true": True, "false": False})
    )
    common = prepared[fixed_15_available.fillna(False)].copy()

    summary = {
        "input_records": int(len(prepared)),
        "common_cohort_records": int(len(common)),
        "excluded_for_fixed_15_comparison": int(len(prepared) - len(common)),
        "positive_records": int(common["target_monta"].sum()),
        "negative_records": int((1 - common["target_monta"]).sum()),
        "positive_prevalence": float(common["target_monta"].mean()),
        "animals": int(common["animal_group"].nunique()),
        "animals_with_positive": int(
            common.loc[common["target_monta"] == 1, "animal_group"].nunique()
        ),
        "collection_year_assumed": int(collection_year),
        "original_date_years": ",".join(
            sorted(
                pd.to_datetime(
                    common["collection_date_original"],
                    format="%d.%m.%Y",
                )
                .dt.year.astype(str)
                .unique()
                .tolist()
            )
        ),
        "date_year_corrected_records": int(
            common["date_year_corrected"].sum()
        ),
        "normalized_collection_dates": int(
            common["collection_date"].nunique()
        ),
        "label_policy": "Monta=true -> 1; Monta vazia -> 0 (provisório)",
        "primary_feature_set": PRIMARY_FEATURE_SET,
        "primary_model": PRIMARY_MODEL,
    }
    return common.reset_index(drop=True), summary


def build_models(random_state: int, rf_estimators: int) -> dict[str, object]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    return {
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=5_000,
                random_state=random_state,
            ),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=rf_estimators,
                min_samples_leaf=3,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=1,
                random_state=random_state,
            ),
        ),
        "svm_rbf": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SVC(
                C=1.0,
                gamma="scale",
                kernel="rbf",
                class_weight="balanced",
            ),
        ),
    }


def prediction_scores(model: object, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features))[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features), dtype=float)
    return np.asarray(model.predict(features), dtype=float)


def calculate_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    score: np.ndarray,
) -> dict[str, float | int]:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    tn, fp, fn, tp = confusion_matrix(
        truth,
        prediction,
        labels=[0, 1],
    ).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    roc_auc = (
        roc_auc_score(truth, score)
        if np.unique(truth).size == 2
        else np.nan
    )
    average_precision = (
        average_precision_score(truth, score)
        if np.unique(truth).size == 2
        else np.nan
    )
    return {
        "roc_auc": float(roc_auc),
        "average_precision": float(average_precision),
        "balanced_accuracy": float(
            balanced_accuracy_score(truth, prediction)
        ),
        "sensitivity": float(recall_score(truth, prediction)),
        "specificity": float(specificity),
        "precision": float(
            precision_score(truth, prediction, zero_division=0)
        ),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
        "mcc": float(matthews_corrcoef(truth, prediction)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def make_splits(
    dataframe: pd.DataFrame,
    *,
    folds: int,
    repeats: int,
    seed: int,
    group_by: str = "animal",
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedGroupKFold

    if group_by not in VALIDATION_GROUP_COLUMNS:
        raise ValueError(
            "Agrupamento inválido: "
            f"{group_by}. Use uma de: {', '.join(VALIDATION_GROUP_COLUMNS)}."
        )

    target = dataframe["target_monta"].to_numpy()
    group_column = VALIDATION_GROUP_COLUMNS[group_by]
    groups = dataframe[group_column].to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < folds:
        raise ValueError(
            f"Há somente {len(unique_groups)} grupos de {group_by} "
            f"para {folds} folds."
        )
    placeholder = np.zeros((len(dataframe), 1))
    splits: list[tuple[int, int, np.ndarray, np.ndarray]] = []

    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(
            n_splits=folds,
            shuffle=True,
            random_state=seed + repeat,
        )
        for fold, (train_index, test_index) in enumerate(
            splitter.split(placeholder, target, groups),
            start=1,
        ):
            train_groups = set(groups[train_index])
            test_groups = set(groups[test_index])
            if train_groups & test_groups:
                raise RuntimeError(
                    f"Vazamento de {group_by} entre treino e teste."
                )
            splits.append((repeat + 1, fold, train_index, test_index))
    return splits


def aggregate_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "roc_auc",
        "average_precision",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "mcc",
    ]
    rows: list[dict[str, object]] = []
    for (feature_set, model), group in fold_metrics.groupby(
        ["feature_set", "model"],
        sort=False,
    ):
        row: dict[str, object] = {
            "feature_set": feature_set,
            "model": model,
            "role": (
                "primary"
                if feature_set == PRIMARY_FEATURE_SET
                and model == PRIMARY_MODEL
                else "comparison"
            ),
            "fold_results": int(len(group)),
        }
        for metric in metric_columns:
            values = group[metric].dropna()
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_p025"] = float(values.quantile(0.025))
            row[f"{metric}_p975"] = float(values.quantile(0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_cluster_bootstrap(
    predictions: pd.DataFrame,
    *,
    iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compara representações reamostrando a unidade de validação."""

    from sklearn.metrics import average_precision_score, roc_auc_score

    logistic = predictions[predictions["model"] == PRIMARY_MODEL]
    averaged = (
        logistic.groupby(
            [
                "feature_set",
                "source_csv_row",
                "id",
                "validation_group_value",
                "target_monta",
            ],
            as_index=False,
        )["score"]
        .mean()
    )
    wide = (
        averaged.pivot(
            index=[
                "source_csv_row",
                "id",
                "validation_group_value",
                "target_monta",
            ],
            columns="feature_set",
            values="score",
        )
        .reset_index()
    )
    groups = wide["validation_group_value"].astype("string").to_numpy()
    truth = wide["target_monta"].to_numpy(dtype=int)
    unique_groups = np.unique(groups)
    group_indices = {
        group: np.flatnonzero(groups == group) for group in unique_groups
    }
    random = np.random.default_rng(seed)
    bootstrap_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    primary_score = wide[PRIMARY_FEATURE_SET].to_numpy(dtype=float)
    for reference in FEATURE_SETS:
        if reference == PRIMARY_FEATURE_SET:
            continue
        reference_score = wide[reference].to_numpy(dtype=float)
        observed = {
            "roc_auc": (
                roc_auc_score(truth, primary_score),
                roc_auc_score(truth, reference_score),
            ),
            "average_precision": (
                average_precision_score(truth, primary_score),
                average_precision_score(truth, reference_score),
            ),
        }
        valid_differences: dict[str, list[float]] = {
            "roc_auc": [],
            "average_precision": [],
        }

        for iteration in range(1, iterations + 1):
            sampled_groups = random.choice(
                unique_groups,
                size=len(unique_groups),
                replace=True,
            )
            sampled_indices = np.concatenate(
                [group_indices[group] for group in sampled_groups]
            )
            sampled_truth = truth[sampled_indices]
            if np.unique(sampled_truth).size < 2:
                continue

            primary_roc = roc_auc_score(
                sampled_truth,
                primary_score[sampled_indices],
            )
            reference_roc = roc_auc_score(
                sampled_truth,
                reference_score[sampled_indices],
            )
            primary_ap = average_precision_score(
                sampled_truth,
                primary_score[sampled_indices],
            )
            reference_ap = average_precision_score(
                sampled_truth,
                reference_score[sampled_indices],
            )
            valid_differences["roc_auc"].append(primary_roc - reference_roc)
            valid_differences["average_precision"].append(
                primary_ap - reference_ap
            )
            bootstrap_rows.append(
                {
                    "reference_feature_set": reference,
                    "iteration": iteration,
                    "roc_auc_difference": primary_roc - reference_roc,
                    "average_precision_difference": (
                        primary_ap - reference_ap
                    ),
                }
            )

        for metric, differences in valid_differences.items():
            difference_array = np.asarray(differences, dtype=float)
            primary_observed, reference_observed = observed[metric]
            summary_rows.append(
                {
                    "primary_feature_set": PRIMARY_FEATURE_SET,
                    "reference_feature_set": reference,
                    "model": PRIMARY_MODEL,
                    "metric": metric,
                    "observed_primary": float(primary_observed),
                    "observed_reference": float(reference_observed),
                    "observed_difference": float(
                        primary_observed - reference_observed
                    ),
                    "bootstrap_iterations": int(len(difference_array)),
                    "difference_mean": float(difference_array.mean()),
                    "difference_p025": float(
                        np.quantile(difference_array, 0.025)
                    ),
                    "difference_p975": float(
                        np.quantile(difference_array, 0.975)
                    ),
                    "fraction_difference_above_zero": float(
                        np.mean(difference_array > 0)
                    ),
                }
            )

    return pd.DataFrame(bootstrap_rows), pd.DataFrame(summary_rows)


def save_primary_plots(
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        PrecisionRecallDisplay,
        RocCurveDisplay,
        confusion_matrix,
    )

    primary = predictions[
        (predictions["feature_set"] == PRIMARY_FEATURE_SET)
        & (predictions["model"] == PRIMARY_MODEL)
    ]
    averaged = (
        primary.groupby("source_csv_row", as_index=False)
        .agg(
            target_monta=("target_monta", "first"),
            score=("score", "mean"),
            predicted_positive_rate=("prediction", "mean"),
        )
    )
    averaged["prediction"] = (
        averaged["predicted_positive_rate"] >= 0.5
    ).astype(int)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    RocCurveDisplay.from_predictions(
        averaged["target_monta"],
        averaged["score"],
        ax=axes[0],
        name="11x11 + regressão logística",
    )
    PrecisionRecallDisplay.from_predictions(
        averaged["target_monta"],
        averaged["score"],
        ax=axes[1],
        name="11x11 + regressão logística",
    )
    ConfusionMatrixDisplay(
        confusion_matrix(
            averaged["target_monta"],
            averaged["prediction"],
            labels=[0, 1],
        ),
        display_labels=["não estro", "estro"],
    ).plot(ax=axes[2], colorbar=False)
    figure.suptitle(
        "Predições fora da amostra — média das repetições por registro"
    )
    figure.tight_layout()
    figure.savefig(
        output_dir / "primary_oof_diagnostics.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    plot_metrics = [
        ("average_precision_mean", "PR-AUC"),
        ("roc_auc_mean", "ROC-AUC"),
        ("f1_mean", "F1"),
    ]
    logistic_summary = summary[
        summary["model"] == PRIMARY_MODEL
    ].set_index("feature_set")
    order = [name for name in FEATURE_SETS if name in logistic_summary.index]
    x_positions = np.arange(len(order))
    width = 0.24
    figure, axis = plt.subplots(figsize=(12, 5.5))
    for metric_index, (column, label) in enumerate(plot_metrics):
        axis.bar(
            x_positions + (metric_index - 1) * width,
            logistic_summary.loc[order, column],
            width=width,
            label=label,
        )
    axis.set_xticks(x_positions)
    axis.set_xticklabels(order, rotation=25, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Média nos folds")
    axis.set_title("Comparação das representações — regressão logística")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(
        output_dir / "feature_set_comparison.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def process(args: argparse.Namespace) -> int:
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_path, sep=";", encoding="utf-8-sig")
    dataframe, dataset_summary = prepare_dataset(
        raw,
        collection_year=args.collection_year,
    )
    splits = make_splits(
        dataframe,
        folds=args.folds,
        repeats=args.repeats,
        seed=args.seed,
        group_by=args.group_by,
    )

    target = dataframe["target_monta"].to_numpy()
    validation_group_column = VALIDATION_GROUP_COLUMNS[args.group_by]
    groups = dataframe[validation_group_column].to_numpy()
    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    assignment_rows: list[dict[str, object]] = []

    for repeat, fold, train_index, test_index in splits:
        train_groups = sorted(set(groups[train_index]))
        test_groups = sorted(set(groups[test_index]))
        assignment_rows.extend(
            {
                "repeat": repeat,
                "fold": fold,
                "validation_group_type": args.group_by,
                "validation_group_value": group,
                "partition": "test",
            }
            for group in test_groups
        )

        model_seed = args.seed + repeat * 100 + fold
        for feature_set, columns in FEATURE_SETS.items():
            train_features = dataframe.loc[train_index, list(columns)]
            test_features = dataframe.loc[test_index, list(columns)]
            for model_name, model in build_models(
                model_seed,
                args.rf_estimators,
            ).items():
                model.fit(train_features, target[train_index])
                prediction = np.asarray(
                    model.predict(test_features),
                    dtype=int,
                )
                score = prediction_scores(model, test_features)
                metrics = calculate_metrics(
                    target[test_index],
                    prediction,
                    score,
                )
                fold_rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "feature_set": feature_set,
                        "model": model_name,
                        "train_records": int(len(train_index)),
                        "test_records": int(len(test_index)),
                        "validation_group_type": args.group_by,
                        "train_groups": int(len(train_groups)),
                        "test_groups": int(len(test_groups)),
                        "train_unique_animals": int(
                            dataframe.loc[
                                train_index, "animal_group"
                            ].nunique()
                        ),
                        "test_unique_animals": int(
                            dataframe.loc[
                                test_index, "animal_group"
                            ].nunique()
                        ),
                        "train_unique_dates": int(
                            dataframe.loc[
                                train_index, "collection_date"
                            ].nunique()
                        ),
                        "test_unique_dates": int(
                            dataframe.loc[
                                test_index, "collection_date"
                            ].nunique()
                        ),
                        "train_positives": int(target[train_index].sum()),
                        "test_positives": int(target[test_index].sum()),
                        **metrics,
                    }
                )
                for local_position, row_index in enumerate(test_index):
                    source = dataframe.iloc[row_index]
                    prediction_rows.append(
                        {
                            "repeat": repeat,
                            "fold": fold,
                            "feature_set": feature_set,
                            "model": model_name,
                            "source_csv_row": int(source["source_csv_row"]),
                            "id": source["animal_group"],
                            "Data": source.get("Data"),
                            "collection_date": source["collection_date"],
                            "Foto": source.get("Foto"),
                            "validation_group_type": args.group_by,
                            "validation_group_value": source[
                                validation_group_column
                            ],
                            "target_monta": int(target[row_index]),
                            "score": float(score[local_position]),
                            "prediction": int(prediction[local_position]),
                        }
                    )

    fold_metrics = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = aggregate_metrics(fold_metrics)
    assignments = pd.DataFrame(assignment_rows)
    feature_definitions = pd.DataFrame(
        [
            {
                "feature_set": feature_set,
                "role": (
                    "primary"
                    if feature_set == PRIMARY_FEATURE_SET
                    else "comparison"
                ),
                "columns": ",".join(columns),
            }
            for feature_set, columns in FEATURE_SETS.items()
        ]
    )
    numeric_feature_columns = sorted(
        {
            column
            for columns in FEATURE_SETS.values()
            for column in columns
        }
    )
    feature_missingness = pd.DataFrame(
        [
            {
                "column": column,
                "missing_records": int(dataframe[column].isna().sum()),
                "missing_fraction": float(dataframe[column].isna().mean()),
            }
            for column in numeric_feature_columns
        ]
    )
    date_audit = (
        dataframe.groupby(
            [
                "collection_date_original",
                "collection_date",
                "date_year_corrected",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            records=("source_csv_row", "size"),
            positives=("target_monta", "sum"),
            animals=("animal_group", "nunique"),
        )
        .sort_values(
            ["collection_date", "collection_date_original"],
            kind="stable",
        )
    )
    bootstrap_rows, bootstrap_summary = paired_cluster_bootstrap(
        predictions,
        iterations=args.bootstrap_iterations,
        seed=args.seed + 10_000,
    )
    summary_table = pd.DataFrame(
        [
            {"item": key, "value": value}
            for key, value in dataset_summary.items()
        ]
        + [
            {"item": "folds", "value": args.folds},
            {"item": "repeats", "value": args.repeats},
            {"item": "validation_group_type", "value": args.group_by},
            {
                "item": "validation_group_column",
                "value": validation_group_column,
            },
            {"item": "random_seed", "value": args.seed},
            {"item": "random_forest_estimators", "value": args.rf_estimators},
            {
                "item": "cluster_bootstrap_iterations",
                "value": args.bootstrap_iterations,
            },
        ]
    )

    outputs = {
        "fold_metrics.csv": fold_metrics,
        "summary_metrics.csv": summary,
        "out_of_fold_predictions.csv": predictions,
        "fold_assignments.csv": assignments,
        "feature_definitions.csv": feature_definitions,
        "feature_missingness.csv": feature_missingness,
        "date_audit.csv": date_audit,
        "dataset_summary.csv": summary_table,
        "paired_bootstrap_differences.csv": bootstrap_rows,
        "paired_bootstrap_summary.csv": bootstrap_summary,
    }
    for filename, output_dataframe in outputs.items():
        output_dataframe.to_csv(
            output_dir / filename,
            sep=";",
            index=False,
            encoding="utf-8-sig",
        )

    save_primary_plots(predictions, summary, output_dir)

    primary = summary[
        (summary["feature_set"] == PRIMARY_FEATURE_SET)
        & (summary["model"] == PRIMARY_MODEL)
    ].iloc[0]
    print(f"Entrada: {input_path}")
    print(f"Coorte comum: {len(dataframe)} registros")
    print(
        f"Classe positiva: {int(dataframe['target_monta'].sum())} "
        f"({dataframe['target_monta'].mean():.1%})"
    )
    print(f"Animais: {dataframe['animal_group'].nunique()}")
    print(
        f"Validação: {args.repeats} repetições x {args.folds} folds "
        f"estratificados e agrupados por {args.group_by}"
    )
    print(
        "Datas com ano corrigido somente na cópia analítica: "
        f"{int(dataframe['date_year_corrected'].sum())}"
    )
    print(
        "Principal 11x11 + regressão logística: "
        f"PR-AUC={primary['average_precision_mean']:.3f}; "
        f"ROC-AUC={primary['roc_auc_mean']:.3f}; "
        f"F1={primary['f1_mean']:.3f}"
    )
    print(f"Saídas: {output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compara representações térmicas com validação agrupada."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--group-by",
        choices=tuple(VALIDATION_GROUP_COLUMNS),
        default="animal",
        help=(
            "Unidade mantida integralmente em treino ou teste: "
            "animal ou data normalizada."
        ),
    )
    parser.add_argument(
        "--collection-year",
        type=int,
        default=2025,
        help=(
            "Ano verdadeiro da coleta usado para corrigir, somente na cópia "
            "analítica, os anos incrementais da planilha."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rf-estimators", type=int, default=300)
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    return parser


def main() -> int:
    try:
        return process(build_parser().parse_args())
    except Exception as error:
        print(f"Erro: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
