"""Avalia detecção no dia e alerta de monta em uma janela de 24 horas.

O protocolo evita três fontes comuns de otimismo:

* separa animais ou datas inteiras entre treino e teste;
* ao validar por data, remove do treino as datas adjacentes às datas de teste,
  porque o alvo de 24 h usa o dia seguinte;
* escolhe o limiar de decisão para F1 somente em validação interna.

As representações de ROI são comparações exploratórias. Sem máscaras manuais
da vulva, regiões térmicas conectadas não devem ser chamadas de segmentação
anatômica.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter
import warnings

import numpy as np
import pandas as pd

from extrair_roi import PROJECT_ROOT


DEFAULT_INPUT = (
    PROJECT_ROOT
    / "outputs"
    / "roi_multiescala"
    / "roi_features_comparative.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modeling_window_24h"
DEFAULT_DATA_DIR = PROJECT_ROOT.parent / "data" / "data"
TARGET_COLUMNS = {
    "mount_today": "target_mount_today",
    "mount_today_or_next_day": "target_mount_today_or_next_day",
    "mount_next_day": "target_mount_next_day",
    "mount_within_24h": "target_mount_within_24h",
}
DEFAULT_TARGETS = ("mount_today", "mount_today_or_next_day")
GROUP_COLUMNS = {
    "animal": "animal_group",
    "date": "collection_date",
}
MODEL_NAMES = ("logistic", "svm_rbf", "extra_trees")
FEATURE_SET_NAMES = (
    "ambient_only",
    "poi",
    "fixed_15",
    "ellipse_4x7",
    "hot_contrast",
    "connected_tight",
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object


def encode_boolean(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin({"true", "1", "sim", "yes"}).astype("int8")


def normalize_collection_dates(
    series: pd.Series,
    *,
    collection_year: int,
) -> pd.Series:
    parsed = pd.to_datetime(
        series.astype("string").str.strip(),
        format="%d.%m.%Y",
        errors="coerce",
    )
    if parsed.isna().any():
        invalid = series.loc[parsed.isna()].astype(str).unique().tolist()
        raise ValueError("Datas inválidas: " + ", ".join(sorted(invalid)))
    return pd.to_datetime(
        parsed.dt.strftime(f"%d.%m.{collection_year}"),
        format="%d.%m.%Y",
        errors="raise",
    )


def make_window_target(
    dataframe: pd.DataFrame,
    base_target: pd.Series,
    *,
    future_days: int,
    start_offset: int = 0,
) -> pd.Series:
    """Marca a observação se houver evento no dia atual ou nos dias seguintes."""

    if start_offset < 0:
        raise ValueError("O deslocamento inicial não pode ser negativo.")
    if future_days < start_offset:
        raise ValueError(
            "O último dia da janela deve ser igual ou posterior ao primeiro."
        )
    positive_events = {
        (str(animal), date)
        for animal, date, target in zip(
            dataframe["animal_group"],
            dataframe["collection_datetime"],
            base_target,
            strict=True,
        )
        if int(target) == 1
    }
    result = []
    for animal, date in zip(
        dataframe["animal_group"],
        dataframe["collection_datetime"],
        strict=True,
    ):
        result.append(
            int(
                any(
                    (str(animal), date + pd.Timedelta(days=offset))
                    in positive_events
                    for offset in range(start_offset, future_days + 1)
                )
            )
        )
    return pd.Series(result, index=dataframe.index, dtype="int8")


def make_timed_window_target(
    dataframe: pd.DataFrame,
    base_target: pd.Series,
    *,
    horizon_hours: float,
) -> pd.Series:
    """Marca eventos estritamente futuros dentro de uma janela cronometrada."""

    if horizon_hours <= 0:
        raise ValueError("A janela em horas deve ser positiva.")
    positive_events: dict[str, list[pd.Timestamp]] = {}
    for animal, timestamp, target in zip(
        dataframe["animal_group"],
        dataframe["collection_datetime"],
        base_target,
        strict=True,
    ):
        if int(target) == 1:
            positive_events.setdefault(str(animal), []).append(
                pd.Timestamp(timestamp)
            )

    horizon = pd.Timedelta(hours=horizon_hours)
    result = []
    for animal, timestamp in zip(
        dataframe["animal_group"],
        dataframe["collection_datetime"],
        strict=True,
    ):
        start = pd.Timestamp(timestamp)
        end = start + horizon
        result.append(
            int(
                any(
                    start < event_timestamp <= end
                    for event_timestamp in positive_events.get(
                        str(animal),
                        [],
                    )
                )
            )
        )
    return pd.Series(result, index=dataframe.index, dtype="int8")


def apply_exif_collection_dates(
    dataframe: pd.DataFrame,
    timestamp_audit_path: Path,
) -> pd.DataFrame:
    """Substitui a data analítica pela data EXIF quando ela está disponível."""

    audit = pd.read_csv(
        timestamp_audit_path,
        sep=";",
        encoding="utf-8-sig",
    )
    required = {"source_csv_row", "exif_date", "timestamp_status"}
    missing = required.difference(audit.columns)
    if missing:
        raise ValueError(
            "Auditoria EXIF sem colunas obrigatórias: "
            + ", ".join(sorted(missing))
        )
    audit_columns = ["source_csv_row", "exif_date", "timestamp_status"]
    if "exif_datetime" in audit.columns:
        audit_columns.append("exif_datetime")
    audit = audit[audit_columns].copy()
    audit["source_csv_row"] = audit["source_csv_row"].astype(int)
    if audit["source_csv_row"].duplicated().any():
        raise ValueError("A auditoria EXIF contém linhas duplicadas da planilha.")

    merged = dataframe.merge(
        audit,
        on="source_csv_row",
        how="left",
        validate="one_to_one",
    )
    exif_dates = pd.to_datetime(merged["exif_date"], errors="coerce")
    if "exif_datetime" in merged.columns:
        exif_datetimes = pd.to_datetime(
            merged["exif_datetime"],
            errors="coerce",
        )
    else:
        exif_datetimes = exif_dates
    verified_datetimes = exif_datetimes.fillna(exif_dates)
    merged["collection_datetime_sheet"] = merged["collection_datetime"]
    merged["collection_datetime"] = verified_datetimes.fillna(
        merged["collection_datetime"]
    )
    merged["collection_date_source"] = np.where(
        verified_datetimes.notna(),
        "exif",
        "sheet_normalized",
    )
    merged["collection_date_exif_changed"] = (
        exif_dates.notna()
        & (
            exif_dates.dt.normalize()
            != merged["collection_datetime_sheet"].dt.normalize()
        )
    ).astype("int8")
    return merged


def summarize(values: np.ndarray, prefix: str) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_p90": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_std": np.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_median": float(np.median(finite)),
        f"{prefix}_p90": float(np.percentile(finite, 90)),
        f"{prefix}_max": float(np.max(finite)),
        f"{prefix}_std": float(np.std(finite)),
    }


def connected_temperature_band(
    matrix: np.ndarray,
    *,
    seed_x: int,
    seed_y: int,
    radius: int = 7,
    tolerance_c: float = 0.1,
) -> tuple[np.ndarray, bool]:
    """Cresce uma região conectada dentro de uma busca espacial limitada."""

    height, width = matrix.shape
    y0 = max(0, seed_y - radius)
    y1 = min(height, seed_y + radius + 1)
    x0 = max(0, seed_x - radius)
    x1 = min(width, seed_x + radius + 1)
    reference_window = matrix[
        max(0, seed_y - 1) : min(height, seed_y + 2),
        max(0, seed_x - 1) : min(width, seed_x + 2),
    ]
    reference = float(np.nanmedian(reference_window))
    allowed = (
        np.isfinite(matrix)
        & (matrix >= reference - tolerance_c)
        & (matrix <= reference + tolerance_c)
    )
    mask = np.zeros_like(matrix, dtype=bool)
    visited = np.zeros_like(matrix, dtype=bool)
    queue: deque[tuple[int, int]] = deque([(seed_y, seed_x)])
    visited[seed_y, seed_x] = True
    mask[seed_y, seed_x] = True
    directions = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    )

    while queue:
        y, x = queue.popleft()
        for delta_y, delta_x in directions:
            next_y = y + delta_y
            next_x = x + delta_x
            if not (y0 <= next_y < y1 and x0 <= next_x < x1):
                continue
            if visited[next_y, next_x]:
                continue
            visited[next_y, next_x] = True
            if allowed[next_y, next_x]:
                mask[next_y, next_x] = True
                queue.append((next_y, next_x))

    touches_limit = bool(
        mask[y0, x0:x1].any()
        or mask[y1 - 1, x0:x1].any()
        or mask[y0:y1, x0].any()
        or mask[y0:y1, x1 - 1].any()
    )
    return mask, touches_limit


def resolve_matrix_path(value: object, data_dir: Path) -> Path:
    matrix_path = Path(str(value))
    if not matrix_path.is_absolute():
        matrix_path = data_dir / matrix_path
    return matrix_path


def extract_candidate_features(
    row: pd.Series,
    *,
    data_dir: Path,
) -> dict[str, float]:
    matrix_path = resolve_matrix_path(row["matrix_path"], data_dir)
    matrix = np.load(matrix_path, allow_pickle=False)
    if matrix.shape != (60, 80):
        raise ValueError(f"Matriz inesperada {matrix.shape}: {matrix_path}")
    seed_x = int(row["seed_x"])
    seed_y = int(row["seed_y"])
    yy, xx = np.ogrid[: matrix.shape[0], : matrix.shape[1]]
    distance = np.sqrt((xx - seed_x) ** 2 + (yy - seed_y) ** 2)
    result: dict[str, float] = {}

    ellipse_mask = (
        ((xx - seed_x) / 4.0) ** 2 + ((yy - seed_y) / 7.0) ** 2 <= 1
    )
    result.update(summarize(matrix[ellipse_mask], "ellipse_4x7"))

    square_15 = matrix[
        seed_y - 7 : seed_y + 8,
        seed_x - 7 : seed_x + 8,
    ]
    if square_15.shape != (15, 15):
        raise ValueError(
            f"ROI 15x15 indisponível em ({seed_x},{seed_y}): {matrix_path}"
        )
    hot_threshold = float(np.nanpercentile(square_15, 90))
    hot_values = square_15[
        np.isfinite(square_15) & (square_15 >= hot_threshold)
    ]
    result.update(summarize(hot_values, "hot_top10"))

    core = matrix[distance <= 4]
    background = matrix[(distance > 5) & (distance <= 8)]
    core_median = float(np.nanmedian(core))
    core_p90 = float(np.nanpercentile(core, 90))
    background_median = float(np.nanmedian(background))
    result["contrast_core_median_minus_background"] = (
        core_median - background_median
    )
    result["contrast_core_p90_minus_background"] = (
        core_p90 - background_median
    )
    result["contrast_hot_fraction_03"] = float(
        np.mean(square_15 >= background_median + 0.3)
    )
    result["contrast_hot_fraction_05"] = float(
        np.mean(square_15 >= background_median + 0.5)
    )

    connected_mask, touches_limit = connected_temperature_band(
        matrix,
        seed_x=seed_x,
        seed_y=seed_y,
    )
    result.update(
        summarize(matrix[connected_mask], "connected_tight")
    )
    result["connected_tight_pixels"] = float(connected_mask.sum())
    result["connected_tight_fraction"] = float(
        connected_mask.sum() / square_15.size
    )
    result["connected_tight_touches_limit"] = float(touches_limit)
    return result


def prepare_dataset(
    input_path: Path,
    *,
    data_dir: Path,
    collection_year: int,
    feature_cache: Path | None = None,
    timestamp_audit_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    dataframe = pd.read_csv(input_path, sep=";", encoding="utf-8-sig")
    common = dataframe[
        dataframe["fixed_15_available"]
        .astype("string")
        .str.lower()
        .eq("true")
    ].copy()
    common["animal_group"] = (
        common["id"].astype("string").str.replace(r"\.0$", "", regex=True)
    )
    common["collection_datetime"] = normalize_collection_dates(
        common["Data"],
        collection_year=collection_year,
    )
    if timestamp_audit_path is not None:
        common = apply_exif_collection_dates(
            common,
            timestamp_audit_path,
        )
    common["collection_date"] = common[
        "collection_datetime"
    ].dt.strftime("%Y-%m-%d")
    common["target_mount_today"] = encode_boolean(common["Monta"])
    common["target_mount_today_or_next_day"] = make_window_target(
        common,
        common["target_mount_today"],
        future_days=1,
    )
    common["target_mount_next_day"] = make_window_target(
        common,
        common["target_mount_today"],
        start_offset=1,
        future_days=1,
    )
    common["target_mount_within_24h"] = make_timed_window_target(
        common,
        common["target_mount_today"],
        horizon_hours=24,
    )

    cache_matches = False
    if feature_cache is not None and feature_cache.is_file():
        cached = pd.read_csv(feature_cache, sep=";", encoding="utf-8-sig")
        cache_matches = (
            len(cached) == len(common)
            and cached["source_csv_row"].astype(int).tolist()
            == common["source_csv_row"].astype(int).tolist()
        )
        if cache_matches:
            new_columns = [
                column
                for column in cached.columns
                if column != "source_csv_row"
            ]
            common = pd.concat(
                [
                    common.reset_index(drop=True),
                    cached[new_columns].reset_index(drop=True),
                ],
                axis=1,
            )

    if not cache_matches:
        extracted = pd.DataFrame(
            [
                extract_candidate_features(row, data_dir=data_dir)
                for _, row in common.iterrows()
            ]
        )
        common = pd.concat(
            [common.reset_index(drop=True), extracted],
            axis=1,
        )
        if feature_cache is not None:
            feature_cache.parent.mkdir(parents=True, exist_ok=True)
            cache_frame = pd.concat(
                [
                    common[["source_csv_row"]].reset_index(drop=True),
                    extracted.reset_index(drop=True),
                ],
                axis=1,
            )
            cache_frame.to_csv(
                feature_cache,
                sep=";",
                index=False,
                encoding="utf-8-sig",
            )

    ambient = ["ambient_temperature_c"]
    feature_sets = {
        "ambient_only": ambient,
        "poi": ["poi_temperature", *ambient],
        "fixed_15": [
            "fixed_15_temp_mean",
            "fixed_15_temp_median",
            "fixed_15_temp_p90",
            "fixed_15_temp_max",
            "fixed_15_temp_std",
            *ambient,
        ],
        "ellipse_4x7": [
            "ellipse_4x7_mean",
            "ellipse_4x7_median",
            "ellipse_4x7_p90",
            "ellipse_4x7_max",
            "ellipse_4x7_std",
            *ambient,
        ],
        "hot_contrast": [
            "hot_top10_mean",
            "hot_top10_median",
            "hot_top10_p90",
            "hot_top10_max",
            "hot_top10_std",
            "contrast_core_median_minus_background",
            "contrast_core_p90_minus_background",
            "contrast_hot_fraction_03",
            "contrast_hot_fraction_05",
            *ambient,
        ],
        "connected_tight": [
            "connected_tight_mean",
            "connected_tight_median",
            "connected_tight_p90",
            "connected_tight_max",
            "connected_tight_std",
            "connected_tight_pixels",
            "connected_tight_fraction",
            "connected_tight_touches_limit",
            *ambient,
        ],
    }
    return common.reset_index(drop=True), feature_sets


def build_model(name: str, *, seed: int) -> ModelSpec:
    from sklearn.ensemble import ExtraTreesClassifier
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
    elif name == "extra_trees":
        estimator = make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=80,
                min_samples_leaf=4,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=1,
                random_state=seed,
            ),
        )
    else:
        raise ValueError(f"Modelo desconhecido: {name}")
    return ModelSpec(name=name, estimator=estimator)


def prediction_scores(model: object, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features))[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features), dtype=float)
    raise TypeError("O modelo não fornece probabilidade nem função de decisão.")


def purge_adjacent_dates(
    train_index: np.ndarray,
    test_index: np.ndarray,
    dates: pd.Series,
    *,
    purge_days: int,
) -> np.ndarray:
    if purge_days <= 0:
        return train_index
    test_dates = pd.to_datetime(dates.iloc[test_index]).to_numpy(
        dtype="datetime64[D]"
    )
    train_dates = pd.to_datetime(dates.iloc[train_index]).to_numpy(
        dtype="datetime64[D]"
    )
    minimum_distance = np.min(
        np.abs(
            train_dates[:, None].astype("int64")
            - test_dates[None, :].astype("int64")
        ),
        axis=1,
    )
    return train_index[minimum_distance > purge_days]


def make_grouped_splits(
    dataframe: pd.DataFrame,
    target: np.ndarray,
    *,
    grouping: str,
    folds: int,
    repeats: int,
    seed: int,
    purge_days: int,
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedGroupKFold

    group_column = GROUP_COLUMNS[grouping]
    groups = dataframe[group_column].astype(str).to_numpy()
    placeholder = np.zeros((len(dataframe), 1))
    result = []
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
            if grouping == "date":
                train_index = purge_adjacent_dates(
                    train_index,
                    test_index,
                    dataframe["collection_datetime"],
                    purge_days=purge_days,
                )
            if np.unique(target[train_index]).size < 2:
                continue
            if np.unique(target[test_index]).size < 2:
                continue
            result.append((repeat + 1, fold, train_index, test_index))
    return result


def select_f1_threshold(
    truth: np.ndarray,
    score: np.ndarray,
) -> tuple[float, float]:
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(truth, score)
    if thresholds.size == 0:
        return 0.5, 0.0
    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_index = int(np.nanargmax(f1_values))
    return float(thresholds[best_index]), float(f1_values[best_index])


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
    group_column = GROUP_COLUMNS[grouping]
    groups = train_frame[group_column].astype(str).to_numpy()
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
    return {
        "roc_auc": float(roc_auc_score(truth, score)),
        "average_precision": float(average_precision_score(truth, score)),
        "balanced_accuracy": float(
            balanced_accuracy_score(truth, prediction)
        ),
        "precision": float(
            precision_score(truth, prediction, zero_division=0)
        ),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
        "mcc": float(matthews_corrcoef(truth, prediction)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def append_progress(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(message + "\n")


def aggregate_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "roc_auc",
        "average_precision",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "mcc",
        "threshold",
    ]
    grouped = fold_metrics.groupby(
        ["target", "grouping", "feature_set", "model"],
        as_index=False,
    )
    summary = grouped[metric_columns].agg(["mean", "std", "count"])
    summary.columns = [
        "_".join(item for item in column if item)
        for column in summary.columns.to_flat_index()
    ]
    return summary


def evaluate(args: argparse.Namespace) -> int:
    if (
        "mount_within_24h" in args.targets
        and args.timestamp_audit is None
    ):
        raise ValueError(
            "O alvo mount_within_24h exige --timestamp-audit com horários EXIF."
        )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.log"
    progress_path.write_text("", encoding="utf-8")
    feature_cache = output_dir / "candidate_features.csv"
    dataframe, feature_sets = prepare_dataset(
        args.input.expanduser().resolve(),
        data_dir=args.data_dir.expanduser().resolve(),
        collection_year=args.collection_year,
        feature_cache=feature_cache,
        timestamp_audit_path=(
            args.timestamp_audit.expanduser().resolve()
            if args.timestamp_audit is not None
            else None
        ),
    )

    selected_targets = {
        name: TARGET_COLUMNS[name] for name in args.targets
    }
    selected_features = {
        name: feature_sets[name] for name in args.feature_sets
    }
    start_time = perf_counter()
    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    split_audit_rows: list[dict[str, object]] = []
    total_configurations = (
        len(selected_targets)
        * len(args.groupings)
        * len(selected_features)
        * len(args.models)
    )
    completed_configurations = 0

    for target_name, target_column in selected_targets.items():
        target = dataframe[target_column].to_numpy(dtype=int)
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
                split_audit_rows.append(
                    {
                        "target": target_name,
                        "grouping": grouping,
                        "repeat": repeat,
                        "fold": fold,
                        "train_records_after_purge": len(train_index),
                        "test_records": len(test_index),
                        "train_positives": int(target[train_index].sum()),
                        "test_positives": int(target[test_index].sum()),
                        "train_groups": int(
                            dataframe.iloc[train_index][
                                GROUP_COLUMNS[grouping]
                            ].nunique()
                        ),
                        "test_groups": int(
                            dataframe.iloc[test_index][
                                GROUP_COLUMNS[grouping]
                            ].nunique()
                        ),
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
                                seed=(
                                    args.seed
                                    + repeat * 1_000
                                    + fold * 10
                                ),
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
                        test_prediction = (
                            test_score >= threshold
                        ).astype(int)
                        metrics = calculate_metrics(
                            target[test_index],
                            test_prediction,
                            test_score,
                        )
                        fold_rows.append(
                            {
                                "target": target_name,
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
                            prediction_rows.append(
                                {
                                    "target": target_name,
                                    "grouping": grouping,
                                    "repeat": repeat,
                                    "fold": fold,
                                    "feature_set": feature_name,
                                    "model": model_name,
                                    "source_csv_row": int(
                                        dataframe.iloc[index][
                                            "source_csv_row"
                                        ]
                                    ),
                                    "truth": int(target[index]),
                                    "score": float(score),
                                    "threshold": float(threshold),
                                    "prediction": int(prediction),
                                }
                            )
                    completed_configurations += 1
                    elapsed = perf_counter() - start_time
                    append_progress(
                        progress_path,
                        (
                            f"{completed_configurations}/"
                            f"{total_configurations} "
                            f"{target_name} {grouping} "
                            f"{feature_name} {model_name} "
                            f"elapsed_s={elapsed:.1f}"
                        ),
                    )

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    split_audit = pd.DataFrame(split_audit_rows).drop_duplicates()
    summary = aggregate_metrics(folds)
    label_summary = pd.DataFrame(
        [
            {
                "target": target_name,
                "records": len(dataframe),
                "positives": int(dataframe[target_column].sum()),
                "negatives": int(
                    len(dataframe) - dataframe[target_column].sum()
                ),
                "prevalence": float(dataframe[target_column].mean()),
            }
            for target_name, target_column in selected_targets.items()
        ]
    )
    timestamp_columns = [
        column
        for column in (
            "source_csv_row",
            "animal_group",
            "Data",
            "collection_datetime_sheet",
            "collection_datetime",
            "collection_date",
            "collection_date_source",
            "collection_date_exif_changed",
            "timestamp_status",
        )
        if column in dataframe.columns
    ]
    timestamp_frame = dataframe[timestamp_columns].copy()
    duplicate_animal_dates = (
        dataframe.groupby(
            ["animal_group", "collection_date"],
            as_index=False,
        )
        .size()
        .query("size > 1")
    )

    for output, name in (
        (label_summary, "label_summary.csv"),
        (timestamp_frame, "dataset_timestamp_audit.csv"),
        (duplicate_animal_dates, "duplicate_animal_dates.csv"),
        (split_audit, "split_audit.csv"),
        (folds, "fold_metrics.csv"),
        (predictions, "out_of_fold_predictions.csv"),
        (summary, "summary_metrics.csv"),
    ):
        output.to_csv(
            output_dir / name,
            sep=";",
            index=False,
            encoding="utf-8-sig",
        )
    append_progress(
        progress_path,
        f"completed elapsed_s={perf_counter() - start_time:.1f}",
    )
    print(label_summary.to_string(index=False))
    print(
        summary.sort_values(
            ["target", "grouping", "average_precision_mean"],
            ascending=[True, True, False],
        )
        .groupby(["target", "grouping"])
        .head(5)
        .to_string(index=False)
    )
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compara o alvo no dia com alerta em 24 h usando validação "
            "agrupada e limiar ajustado dentro do treino."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--collection-year", type=int, default=2025)
    parser.add_argument(
        "--timestamp-audit",
        type=Path,
        help=(
            "CSV produzido por auditar_timestamps_exif.py. Quando informado, "
            "a data EXIF substitui a data normalizada da planilha."
        ),
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--purge-days", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=tuple(TARGET_COLUMNS),
        default=list(DEFAULT_TARGETS),
    )
    parser.add_argument(
        "--groupings",
        nargs="+",
        choices=tuple(GROUP_COLUMNS),
        default=list(GROUP_COLUMNS),
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=FEATURE_SET_NAMES,
        default=list(FEATURE_SET_NAMES),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=["logistic", "svm_rbf"],
    )
    return parser


def main() -> int:
    return evaluate(make_parser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
