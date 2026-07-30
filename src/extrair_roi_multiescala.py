"""Extrai atributos térmicos de ROIs fixas multiescala centradas no POI.

Estratégia pré-especificada:

* ROI principal: janela quadrada 11 x 11 centrada em (Coord_X, Coord_Y);
* sensibilidade espacial: janelas 7 x 7 e 15 x 15;
* comparação exploratória: crescimento de região limitado (raio 5, ±0,3 °C);
* baselines: temperatura radiométrica no POI e leitura de ponto da planilha.

As janelas fixas são o método principal porque suas fronteiras são
determinísticas e reproduzíveis. O region growing é preservado para comparação,
mas não é tratado como máscara anatômica validada.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from extrair_roi import (
    DEFAULT_CSV,
    PROJECT_ROOT,
    extract_temperature_features,
    locate_matrix,
    parse_temperature,
    photo_stem,
    region_growing_from_seed,
    resolve_data_dir,
    safe_path_component,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "roi_multiescala"
DEFAULT_WINDOW_SIZES = (7, 11, 15)
PRIMARY_WINDOW_SIZE = 11


def validate_window_sizes(window_sizes: tuple[int, ...]) -> tuple[int, ...]:
    unique_sizes = tuple(sorted(set(window_sizes)))
    if not unique_sizes:
        raise ValueError("Informe pelo menos um tamanho de janela.")
    for size in unique_sizes:
        if size <= 0 or size % 2 == 0:
            raise ValueError(
                f"O tamanho {size} é inválido; use inteiros positivos ímpares."
            )
    return unique_sizes


def fixed_square_mask(
    matrix_shape: tuple[int, int],
    seed_y: int,
    seed_x: int,
    size: int,
) -> np.ndarray:
    """Retorna uma janela quadrada de tamanho ímpar centrada na seed."""

    validate_window_sizes((size,))
    height, width = matrix_shape
    radius = size // 2
    x0 = seed_x - radius
    x1 = seed_x + radius + 1
    y0 = seed_y - radius
    y1 = seed_y + radius + 1
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        raise ValueError(
            f"Janela {size}x{size} centrada em ({seed_x},{seed_y}) "
            f"ultrapassa a matriz {width}x{height}."
        )

    mask = np.zeros(matrix_shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def prefixed_features(
    prefix: str,
    matrix: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    features = extract_temperature_features(matrix, mask)
    return {
        f"{prefix}_{name.removeprefix('roi_')}": value
        for name, value in features.items()
    }


def add_ambient_gradients(
    row: dict[str, object],
    prefix: str,
    ambient_temperature: float | None,
) -> None:
    if ambient_temperature is None:
        return
    row[f"{prefix}_mean_minus_ambient"] = (
        float(row[f"{prefix}_temp_mean"]) - ambient_temperature
    )
    row[f"{prefix}_p90_minus_ambient"] = (
        float(row[f"{prefix}_temp_p90"]) - ambient_temperature
    )
    row[f"{prefix}_max_minus_ambient"] = (
        float(row[f"{prefix}_temp_max"]) - ambient_temperature
    )


def thermal_rgb(matrix: np.ndarray) -> np.ndarray:
    finite = matrix[np.isfinite(matrix)]
    minimum = float(finite.min())
    maximum = float(finite.max())
    normalized = np.clip(
        (matrix - minimum) / max(maximum - minimum, np.finfo(float).eps),
        0,
        1,
    )
    positions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    colors = np.array(
        [
            [0, 0, 4],
            [60, 15, 110],
            [160, 45, 128],
            [245, 120, 60],
            [252, 253, 191],
        ],
        dtype=float,
    )
    rgb = np.stack(
        [
            np.interp(normalized, positions, colors[:, channel])
            for channel in range(3)
        ],
        axis=-1,
    )
    return np.clip(rgb, 0, 255).astype(np.uint8)


def save_comparison_overlay(
    matrix: np.ndarray,
    region_mask: np.ndarray,
    destination: Path,
    *,
    seed_x: int,
    seed_y: int,
    window_sizes: tuple[int, ...],
    title: str,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    rgb = thermal_rgb(matrix).astype(float)
    rgb[region_mask] = (
        0.62 * rgb[region_mask] + 0.38 * np.array([255, 0, 255])
    )
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    scale = 8
    thermal_image = Image.fromarray(rgb, mode="RGB").resize(
        (matrix.shape[1] * scale, matrix.shape[0] * scale),
        resample=Image.Resampling.NEAREST,
    )
    title_height = 72
    canvas = Image.new(
        "RGB",
        (thermal_image.width, thermal_image.height + title_height),
        "white",
    )
    canvas.paste(thermal_image, (0, title_height))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.multiline_text((8, 6), title, fill="black", font=font, spacing=3)

    colors = {
        7: "#00FF66",
        11: "#00FFFF",
        15: "#FFFFFF",
    }
    legend = " | ".join(
        f"{size}x{size}={colors.get(size, '#FFFF00')}"
        for size in window_sizes
    )
    draw.text(
        (8, 49),
        f"Janelas: {legend} | RG=magenta",
        fill="black",
        font=font,
    )

    for size in sorted(window_sizes, reverse=True):
        radius = size // 2
        color = colors.get(size, "#FFFF00")
        draw.rectangle(
            (
                (seed_x - radius) * scale,
                title_height + (seed_y - radius) * scale,
                (seed_x + radius + 1) * scale - 1,
                title_height + (seed_y + radius + 1) * scale - 1,
            ),
            outline=color,
            width=2,
        )

    center_x = seed_x * scale + scale // 2
    center_y = title_height + seed_y * scale + scale // 2
    cross_radius = scale
    draw.line(
        (
            center_x - cross_radius,
            center_y,
            center_x + cross_radius,
            center_y,
        ),
        fill="white",
        width=2,
    )
    draw.line(
        (
            center_x,
            center_y - cross_radius,
            center_x,
            center_y + cross_radius,
        ),
        fill="white",
        width=2,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def process_dataset(args: argparse.Namespace) -> int:
    csv_path = args.csv.expanduser().resolve()
    data_dir = resolve_data_dir(args.data_dir)
    output_dir = args.output_dir.expanduser().resolve()
    reviews_dir = output_dir / "revisao"
    output_dir.mkdir(parents=True, exist_ok=True)
    window_sizes = validate_window_sizes(tuple(args.window_sizes))
    if PRIMARY_WINDOW_SIZE not in window_sizes:
        raise ValueError(
            f"A janela principal {PRIMARY_WINDOW_SIZE}x{PRIMARY_WINDOW_SIZE} "
            "deve constar em --window-sizes."
        )
    primary_radius = PRIMARY_WINDOW_SIZE // 2

    dataframe = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
    required_columns = {"id", "Lote", "Foto", "Coord_X", "Coord_Y"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            "Colunas obrigatórias ausentes: " + ", ".join(sorted(missing_columns))
        )

    eligible = dataframe[
        dataframe["Foto"].notna()
        & dataframe["Coord_X"].notna()
        & dataframe["Coord_Y"].notna()
    ]
    if args.limit is not None:
        eligible = eligible.head(args.limit)

    feature_rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    review_payloads: list[dict[str, object]] = []

    for index, source_row in eligible.iterrows():
        try:
            matrix_path, lot_name, animal_name = locate_matrix(
                data_dir,
                source_row["Lote"],
                source_row["id"],
                source_row["Foto"],
            )
            if not matrix_path.is_file():
                raise FileNotFoundError(f"Matriz não encontrada: {matrix_path}")

            matrix = np.load(matrix_path, allow_pickle=False)
            if matrix.shape != (60, 80):
                raise ValueError(
                    f"Matriz com formato inesperado {matrix.shape}: {matrix_path}"
                )

            seed_x = int(round(float(source_row["Coord_X"])))
            seed_y = int(round(float(source_row["Coord_Y"])))
            if seed_x == 0 and seed_y == 0:
                raise ValueError(
                    "POI (0,0) tratado como marcação ausente; requer revisão manual."
                )
            if (
                seed_x < primary_radius
                or seed_x >= matrix.shape[1] - primary_radius
                or seed_y < primary_radius
                or seed_y >= matrix.shape[0] - primary_radius
            ):
                raise ValueError(
                    f"POI ({seed_x},{seed_y}) próximo demais da borda para "
                    f"a janela principal "
                    f"{PRIMARY_WINDOW_SIZE}x{PRIMARY_WINDOW_SIZE}."
                )

            output_row = source_row.to_dict()
            output_row.update(
                {
                    "source_csv_row": int(index) + 2,
                    "matrix_path": str(matrix_path.relative_to(data_dir)),
                    "seed_x": seed_x,
                    "seed_y": seed_y,
                    "poi_temperature": float(matrix[seed_y, seed_x]),
                    "primary_roi_method": f"fixed_square_{PRIMARY_WINDOW_SIZE}",
                    "sensitivity_roi_methods": ",".join(
                        f"fixed_square_{size}"
                        for size in window_sizes
                        if size != PRIMARY_WINDOW_SIZE
                    ),
                    "comparison_roi_method": (
                        f"region_growing_r{args.rg_max_radius}_"
                        f"t{args.rg_tolerance:g}"
                    ),
                }
            )

            ambient_temperature = parse_temperature(
                source_row.get("Temp. Ambiente")
            )
            output_row["ambient_temperature_c"] = ambient_temperature
            if ambient_temperature is not None:
                output_row["poi_minus_ambient"] = (
                    output_row["poi_temperature"] - ambient_temperature
                )

            for size in window_sizes:
                prefix = f"fixed_{size}"
                try:
                    mask = fixed_square_mask(
                        matrix.shape,
                        seed_y,
                        seed_x,
                        size,
                    )
                except ValueError:
                    output_row[f"{prefix}_available"] = False
                else:
                    output_row[f"{prefix}_available"] = True
                    output_row.update(
                        prefixed_features(prefix, matrix, mask)
                    )
                    add_ambient_gradients(
                        output_row,
                        prefix,
                        ambient_temperature,
                    )

            region_result = region_growing_from_seed(
                matrix,
                seed_y,
                seed_x,
                tolerance=args.rg_tolerance,
                max_radius=args.rg_max_radius,
                reference_radius=args.rg_reference_radius,
                connectivity=8,
            )
            output_row.update(
                prefixed_features("rg", matrix, region_result.mask)
            )
            output_row["rg_reference_temperature"] = (
                region_result.reference_temperature
            )
            output_row["rg_tolerance_c"] = args.rg_tolerance
            output_row["rg_max_radius_px"] = args.rg_max_radius
            output_row["rg_search_pixels"] = (
                region_result.search_bounds[2] - region_result.search_bounds[0]
            ) * (
                region_result.search_bounds[3] - region_result.search_bounds[1]
            )
            output_row["rg_area_fraction"] = (
                int(region_result.mask.sum()) / output_row["rg_search_pixels"]
            )
            x0, y0, x1, y1 = region_result.search_bounds
            output_row["rg_touches_limit"] = bool(
                region_result.mask[y0, x0:x1].any()
                or region_result.mask[y1 - 1, x0:x1].any()
                or region_result.mask[y0:y1, x0].any()
                or region_result.mask[y0:y1, x1 - 1].any()
            )
            add_ambient_gradients(output_row, "rg", ambient_temperature)

            feature_rows.append(output_row)
            review_payloads.append(
                {
                    "row": output_row,
                    "matrix": matrix,
                    "region_mask": region_result.mask,
                    "lot_name": lot_name,
                    "animal_name": animal_name,
                    "available_window_sizes": tuple(
                        size
                        for size in window_sizes
                        if output_row[f"fixed_{size}_available"]
                    ),
                }
            )
        except Exception as error:
            errors.append(
                {
                    "source_csv_row": int(index) + 2,
                    "id": source_row.get("id"),
                    "Lote": source_row.get("Lote"),
                    "Foto": source_row.get("Foto"),
                    "erro": str(error),
                }
            )

    features_dataframe = pd.DataFrame(feature_rows)
    features_path = output_dir / "roi_features_comparative.csv"
    features_dataframe.to_csv(
        features_path,
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )
    errors_path = output_dir / "roi_errors.csv"
    pd.DataFrame(errors).to_csv(
        errors_path,
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )

    summary_rows = []
    if not features_dataframe.empty:
        for size in window_sizes:
            prefix = f"fixed_{size}"
            method_dataframe = features_dataframe[
                features_dataframe[f"{prefix}_available"]
            ]
            summary_rows.append(
                {
                    "method": f"fixed_square_{size}",
                    "role": (
                        "primary"
                        if size == PRIMARY_WINDOW_SIZE
                        else "sensitivity"
                    ),
                    "records": len(method_dataframe),
                    "pixels_min": int(
                        method_dataframe[f"{prefix}_pixels"].min()
                    ),
                    "pixels_median": float(
                        method_dataframe[f"{prefix}_pixels"].median()
                    ),
                    "pixels_max": int(
                        method_dataframe[f"{prefix}_pixels"].max()
                    ),
                    "temperature_mean_median": float(
                        method_dataframe[f"{prefix}_temp_mean"].median()
                    ),
                    "temperature_p90_median": float(
                        method_dataframe[f"{prefix}_temp_p90"].median()
                    ),
                }
            )
        summary_rows.append(
            {
                "method": "region_growing",
                "role": "comparison",
                "records": len(features_dataframe),
                "pixels_min": int(features_dataframe["rg_pixels"].min()),
                "pixels_median": float(
                    features_dataframe["rg_pixels"].median()
                ),
                "pixels_max": int(features_dataframe["rg_pixels"].max()),
                "temperature_mean_median": float(
                    features_dataframe["rg_temp_mean"].median()
                ),
                "temperature_p90_median": float(
                    features_dataframe["rg_temp_p90"].median()
                ),
                "touches_limit_count": int(
                    features_dataframe["rg_touches_limit"].sum()
                ),
            }
        )
    summary_path = output_dir / "roi_method_summary.csv"
    pd.DataFrame(summary_rows).to_csv(
        summary_path,
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )

    if reviews_dir.is_dir():
        for previous_overlay in reviews_dir.glob("linha_*.png"):
            previous_overlay.unlink()

    if args.review_count > 0 and review_payloads:
        review_count = min(args.review_count, len(review_payloads))
        positions = sorted(
            set(
                np.linspace(
                    0,
                    len(review_payloads) - 1,
                    num=review_count,
                    dtype=int,
                ).tolist()
            )
        )
        for position in positions:
            payload = review_payloads[position]
            row = payload["row"]
            review_name = (
                f"linha_{int(row['source_csv_row']):04d}_"
                f"{safe_path_component(str(row['Lote']))}_"
                f"{photo_stem(row['Foto'])}.png"
            )
            save_comparison_overlay(
                payload["matrix"],
                payload["region_mask"],
                reviews_dir / review_name,
                seed_x=int(row["seed_x"]),
                seed_y=int(row["seed_y"]),
                window_sizes=payload["available_window_sizes"],
                title=(
                    f"{payload['lot_name']} | {payload['animal_name']} | "
                    f"{photo_stem(row['Foto'])}\n"
                    f"POI=({int(row['seed_x'])},{int(row['seed_y'])}) | "
                    f"RG={int(row['rg_pixels'])} px"
                ),
            )

    print(f"Dados: {data_dir}")
    print(f"Planilha: {csv_path}")
    print(f"Linhas com POI: {len(eligible)}")
    print(f"Registros processados: {len(features_dataframe)}")
    print(f"Erros: {len(errors)}")
    print(
        "Janelas fixas: "
        + ", ".join(f"{size}x{size}" for size in window_sizes)
    )
    print(f"Método principal: {PRIMARY_WINDOW_SIZE}x{PRIMARY_WINDOW_SIZE}")
    print(f"Atributos comparativos: {features_path}")
    print(f"Resumo dos métodos: {summary_path}")
    print(f"Amostras visuais: {reviews_dir}")
    return 0 if feature_rows else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extrai ROIs fixas multiescala e compara com region growing."
        )
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--window-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_WINDOW_SIZES),
    )
    parser.add_argument("--rg-tolerance", type=float, default=0.3)
    parser.add_argument("--rg-max-radius", type=int, default=5)
    parser.add_argument("--rg-reference-radius", type=int, default=1)
    parser.add_argument("--review-count", type=int, default=24)
    parser.add_argument("--limit", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return process_dataset(args)
    except Exception as error:
        print(f"Erro: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
