"""Bootstrap agrupado das predições fora da amostra da janela de 24 h.

As predições das repetições são primeiro promediadas por registro. Depois,
animais ou datas inteiras são reamostrados com reposição, preservando a
dependência entre fotografias do mesmo grupo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from avaliar_janela_24h import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT_DIR,
    apply_exif_collection_dates,
    normalize_collection_dates,
)


METRICS = ("average_precision", "roc_auc")


def metric_value(
    name: str,
    truth: np.ndarray,
    score: np.ndarray,
) -> float:
    from sklearn.metrics import average_precision_score, roc_auc_score

    if name == "average_precision":
        return float(average_precision_score(truth, score))
    if name == "roc_auc":
        return float(roc_auc_score(truth, score))
    raise ValueError(f"Métrica desconhecida: {name}")


def load_averaged_predictions(
    predictions_path: Path,
    metadata_path: Path,
    *,
    target: str,
    model: str,
    collection_year: int,
    timestamp_audit_path: Path | None = None,
) -> pd.DataFrame:
    predictions = pd.read_csv(
        predictions_path,
        sep=";",
        encoding="utf-8-sig",
    )
    predictions = predictions[
        predictions["target"].eq(target)
        & predictions["model"].eq(model)
    ]
    if predictions.empty:
        raise ValueError(
            f"Nenhuma predição para target={target!r}, model={model!r}."
        )
    averaged = (
        predictions.groupby(
            [
                "grouping",
                "feature_set",
                "source_csv_row",
                "truth",
            ],
            as_index=False,
        )["score"]
        .mean()
    )
    metadata = pd.read_csv(
        metadata_path,
        sep=";",
        encoding="utf-8-sig",
        usecols=["source_csv_row", "id", "Data"],
    )
    metadata["animal"] = (
        metadata["id"].astype("string").str.replace(r"\.0$", "", regex=True)
    )
    metadata["collection_datetime"] = normalize_collection_dates(
        metadata["Data"],
        collection_year=collection_year,
    )
    if timestamp_audit_path is not None:
        metadata = apply_exif_collection_dates(
            metadata,
            timestamp_audit_path,
        )
    metadata["date"] = metadata["collection_datetime"].dt.strftime("%Y-%m-%d")
    merged = averaged.merge(
        metadata[["source_csv_row", "animal", "date"]],
        on="source_csv_row",
        how="left",
        validate="many_to_one",
    )
    if merged[["animal", "date"]].isna().any().any():
        raise ValueError("Há predições sem metadados de animal ou data.")
    return merged


def cluster_bootstrap(
    averaged: pd.DataFrame,
    *,
    reference: str,
    iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    random = np.random.default_rng(seed)
    absolute_rows: list[dict[str, object]] = []
    difference_rows: list[dict[str, object]] = []

    for grouping in ("animal", "date"):
        subset = averaged[averaged["grouping"].eq(grouping)]
        if subset.empty:
            continue
        wide = (
            subset.pivot(
                index=["source_csv_row", "truth", grouping],
                columns="feature_set",
                values="score",
            )
            .reset_index()
        )
        if reference not in wide:
            raise ValueError(
                f"Referência {reference!r} ausente no grupo {grouping!r}."
            )
        truth = wide["truth"].to_numpy(dtype=int)
        groups = wide[grouping].astype(str).to_numpy()
        unique_groups = np.unique(groups)
        group_indices = {
            group: np.flatnonzero(groups == group)
            for group in unique_groups
        }
        non_feature_columns = {
            "source_csv_row",
            "truth",
            "animal",
            "date",
        }
        features = [
            column
            for column in wide.columns
            if column not in non_feature_columns
        ]
        bootstrap_metrics = {
            (feature, metric): []
            for feature in features
            for metric in METRICS
        }
        bootstrap_differences = {
            (feature, metric): []
            for feature in features
            if feature != reference
            for metric in METRICS
        }

        for _ in range(iterations):
            sampled_groups = random.choice(
                unique_groups,
                size=len(unique_groups),
                replace=True,
            )
            sampled_index = np.concatenate(
                [group_indices[group] for group in sampled_groups]
            )
            sampled_truth = truth[sampled_index]
            if np.unique(sampled_truth).size < 2:
                continue
            reference_scores = wide[reference].to_numpy(dtype=float)[
                sampled_index
            ]
            for metric in METRICS:
                reference_value = metric_value(
                    metric,
                    sampled_truth,
                    reference_scores,
                )
                for feature in features:
                    feature_scores = wide[feature].to_numpy(dtype=float)[
                        sampled_index
                    ]
                    value = metric_value(
                        metric,
                        sampled_truth,
                        feature_scores,
                    )
                    bootstrap_metrics[(feature, metric)].append(value)
                    if feature != reference:
                        bootstrap_differences[(feature, metric)].append(
                            value - reference_value
                        )

        reference_score = wide[reference].to_numpy(dtype=float)
        reference_observed = {
            metric: metric_value(metric, truth, reference_score)
            for metric in METRICS
        }
        for feature in features:
            feature_score = wide[feature].to_numpy(dtype=float)
            for metric in METRICS:
                values = np.asarray(
                    bootstrap_metrics[(feature, metric)],
                    dtype=float,
                )
                if values.size == 0:
                    raise ValueError(
                        f"Nenhuma reamostragem válida para {grouping}."
                    )
                observed = metric_value(metric, truth, feature_score)
                absolute_rows.append(
                    {
                        "grouping": grouping,
                        "feature_set": feature,
                        "metric": metric,
                        "observed": observed,
                        "bootstrap_p025": float(
                            np.quantile(values, 0.025)
                        ),
                        "bootstrap_p975": float(
                            np.quantile(values, 0.975)
                        ),
                        "valid_iterations": len(values),
                    }
                )
                if feature == reference:
                    continue
                differences = np.asarray(
                    bootstrap_differences[(feature, metric)],
                    dtype=float,
                )
                difference_rows.append(
                    {
                        "grouping": grouping,
                        "feature_set": feature,
                        "reference": reference,
                        "metric": metric,
                        "observed_difference": (
                            observed - reference_observed[metric]
                        ),
                        "bootstrap_p025": float(
                            np.quantile(differences, 0.025)
                        ),
                        "bootstrap_p975": float(
                            np.quantile(differences, 0.975)
                        ),
                        "valid_iterations": len(differences),
                    }
                )

    return pd.DataFrame(absolute_rows), pd.DataFrame(difference_rows)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calcula bootstrap pareado por animal e data para as predições "
            "fora da amostra do experimento de janela de 24 h."
        )
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "out_of_fold_predictions.csv",
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--target",
        default="mount_today_or_next_day",
    )
    parser.add_argument("--model", default="logistic")
    parser.add_argument("--reference", default="fixed_15")
    parser.add_argument("--collection-year", type=int, default=2025)
    parser.add_argument(
        "--timestamp-audit",
        type=Path,
        help="CSV de datas EXIF produzido por auditar_timestamps_exif.py.",
    )
    parser.add_argument("--iterations", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20260730)
    return parser


def main() -> int:
    args = make_parser().parse_args()
    averaged = load_averaged_predictions(
        args.predictions.expanduser().resolve(),
        args.metadata.expanduser().resolve(),
        target=args.target,
        model=args.model,
        collection_year=args.collection_year,
        timestamp_audit_path=(
            args.timestamp_audit.expanduser().resolve()
            if args.timestamp_audit is not None
            else None
        ),
    )
    absolute, differences = cluster_bootstrap(
        averaged,
        reference=args.reference,
        iterations=args.iterations,
        seed=args.seed,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    absolute.to_csv(
        output_dir / "bootstrap_absolute_metrics.csv",
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )
    differences.to_csv(
        output_dir / "bootstrap_roi_differences.csv",
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )
    print(absolute.to_string(index=False))
    print(differences.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
