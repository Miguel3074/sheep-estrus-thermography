"""Extrai ROIs térmicas a partir dos pontos anotados na planilha.

O ponto (Coord_X, Coord_Y) é usado como semente de um crescimento de região
limitado espacialmente. A limitação evita que a máscara "vaze" para partes
quentes do corpo em matrizes térmicas de baixa resolução (80 x 60).
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
import re
import sys
import unicodedata

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "planilha" / "planilha_anotada.CSV"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "roi_seed_r5_t03_final"

# Identificadores usados na planilha para animais sem número de brinco legível.
ANIMAL_ALIASES = {
    "2173": "Ovelha sem Brinco - ♡",
    "2191": "Ovelha Brinco Quebrado",
}


@dataclass(frozen=True)
class RegionGrowingResult:
    mask: np.ndarray
    reference_temperature: float
    search_bounds: tuple[int, int, int, int]


def region_growing_from_seed(
    matrix: np.ndarray,
    seed_y: int,
    seed_x: int,
    *,
    tolerance: float = 0.3,
    max_radius: int = 5,
    reference_radius: int = 1,
    connectivity: int = 8,
) -> RegionGrowingResult:
    """Cria uma máscara conectada ao redor da semente.

    A temperatura de referência é a mediana de uma janela local ao redor da
    semente. Um pixel entra na ROI quando:

    1. está conectado à semente;
    2. está dentro do raio espacial máximo;
    3. difere da referência em no máximo ``tolerance`` graus Celsius.
    """

    if matrix.ndim != 2:
        raise ValueError(f"A matriz deve ser bidimensional; recebido {matrix.shape}.")
    if tolerance < 0:
        raise ValueError("A tolerância térmica não pode ser negativa.")
    if max_radius < 0 or reference_radius < 0:
        raise ValueError("Os raios espacial e de referência não podem ser negativos.")
    if connectivity not in (4, 8):
        raise ValueError("A conectividade deve ser 4 ou 8.")

    height, width = matrix.shape
    if not (0 <= seed_y < height and 0 <= seed_x < width):
        raise ValueError(
            f"Semente ({seed_x}, {seed_y}) fora da matriz {width}x{height}."
        )
    if not np.isfinite(matrix[seed_y, seed_x]):
        raise ValueError("A temperatura da semente não é finita.")

    ref_y0 = max(0, seed_y - reference_radius)
    ref_y1 = min(height, seed_y + reference_radius + 1)
    ref_x0 = max(0, seed_x - reference_radius)
    ref_x1 = min(width, seed_x + reference_radius + 1)
    reference_window = matrix[ref_y0:ref_y1, ref_x0:ref_x1]
    reference_temperature = float(
        np.median(reference_window[np.isfinite(reference_window)])
    )

    y0 = max(0, seed_y - max_radius)
    y1 = min(height, seed_y + max_radius + 1)
    x0 = max(0, seed_x - max_radius)
    x1 = min(width, seed_x + max_radius + 1)

    lower = reference_temperature - tolerance
    upper = reference_temperature + tolerance
    allowed = np.isfinite(matrix) & (matrix >= lower) & (matrix <= upper)

    if connectivity == 4:
        directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
    else:
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

    mask = np.zeros((height, width), dtype=bool)
    visited = np.zeros((height, width), dtype=bool)
    queue: deque[tuple[int, int]] = deque([(seed_y, seed_x)])
    visited[seed_y, seed_x] = True
    mask[seed_y, seed_x] = True

    while queue:
        y, x = queue.popleft()
        for delta_y, delta_x in directions:
            next_y = y + delta_y
            next_x = x + delta_x
            inside_search_area = y0 <= next_y < y1 and x0 <= next_x < x1
            if not inside_search_area or visited[next_y, next_x]:
                continue
            visited[next_y, next_x] = True
            if allowed[next_y, next_x]:
                mask[next_y, next_x] = True
                queue.append((next_y, next_x))

    return RegionGrowingResult(
        mask=mask,
        reference_temperature=reference_temperature,
        search_bounds=(x0, y0, x1, y1),
    )


def extract_temperature_features(
    matrix: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    """Resume as temperaturas contidas na ROI."""

    values = matrix[mask & np.isfinite(matrix)]
    if values.size == 0:
        raise ValueError("A máscara não contém temperaturas válidas.")

    return {
        "roi_pixels": int(values.size),
        "roi_temp_min": float(np.min(values)),
        "roi_temp_mean": float(np.mean(values)),
        "roi_temp_median": float(np.median(values)),
        "roi_temp_p90": float(np.percentile(values, 90)),
        "roi_temp_max": float(np.max(values)),
        "roi_temp_std": float(np.std(values)),
    }


def resolve_data_dir(explicit_path: Path | None) -> Path:
    """Localiza a pasta que contém os lotes e as matrizes."""

    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    environment_path = os.environ.get("SHEEP_DATA_DIR")
    if environment_path:
        candidates.append(Path(environment_path))
    candidates.extend(
        [
            PROJECT_ROOT / "data",
            PROJECT_ROOT.parent / "data" / "data",
        ]
    )

    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "Lote Rosa").is_dir() and (
            resolved / "Lote Vermelho"
        ).is_dir():
            return resolved

    attempted = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Não foi encontrada uma pasta de dados com os dois lotes. "
        "Use --data-dir ou SHEEP_DATA_DIR.\n"
        f"Caminhos verificados:\n{attempted}"
    )


def normalize_identifier(value: object) -> str:
    """Converte IDs numéricos vindos do pandas para nomes de pasta estáveis."""

    text = str(value).strip()
    try:
        numeric = float(text)
        if numeric.is_integer():
            return str(int(numeric))
    except ValueError:
        pass
    return text


def animal_directory_name(identifier: object) -> str:
    normalized = normalize_identifier(identifier)
    return ANIMAL_ALIASES.get(normalized, f"Ovelha {normalized}")


def photo_stem(value: object) -> str:
    return f"FLIR{int(float(value)):04d}"


def parse_temperature(value: object) -> float | None:
    if pd.isna(value):
        return None
    match = re.search(r"-?\d+(?:[.,]\d+)?", str(value))
    if match is None:
        return None
    return float(match.group(0).replace(",", "."))


def safe_path_component(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_value)
    return cleaned.strip("_") or "sem_nome"


def locate_matrix(
    data_dir: Path,
    lot_value: object,
    animal_id: object,
    photo_value: object,
) -> tuple[Path, str, str]:
    lot_name = f"Lote {str(lot_value).strip()}"
    animal_name = animal_directory_name(animal_id)
    stem = photo_stem(photo_value)
    matrix_path = data_dir / lot_name / animal_name / f"{stem}.npy"
    return matrix_path, lot_name, animal_name


def mask_quality_flags(
    mask: np.ndarray,
    search_bounds: tuple[int, int, int, int],
    *,
    min_pixels: int,
    large_fraction: float,
    search_clipped: bool,
) -> tuple[str, float, bool]:
    x0, y0, x1, y1 = search_bounds
    search_area = max(1, (x1 - x0) * (y1 - y0))
    fraction = float(mask.sum() / search_area)
    touches_limit = bool(
        mask[y0, x0:x1].any()
        or mask[y1 - 1, x0:x1].any()
        or mask[y0:y1, x0].any()
        or mask[y0:y1, x1 - 1].any()
    )

    flags = []
    if int(mask.sum()) < min_pixels:
        flags.append("ROI_PEQUENA")
    if fraction >= large_fraction:
        flags.append("ROI_GRANDE")
    if touches_limit:
        flags.append("TOCA_LIMITE")
    if search_clipped:
        flags.append("POI_PROXIMO_BORDA")
    return ("OK" if not flags else "|".join(flags), fraction, touches_limit)


def save_review_overlay(
    matrix_path: Path,
    mask_path: Path,
    destination: Path,
    *,
    seed_x: int,
    seed_y: int,
    search_bounds: tuple[int, int, int, int],
    title: str,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    matrix = np.load(matrix_path, allow_pickle=False)
    mask = np.load(mask_path, allow_pickle=False).astype(bool)
    x0, y0, x1, y1 = search_bounds

    finite = matrix[np.isfinite(matrix)]
    minimum = float(finite.min())
    maximum = float(finite.max())
    normalized = np.clip(
        (matrix - minimum) / max(maximum - minimum, np.finfo(float).eps),
        0,
        1,
    )

    # Gradiente inspirado no "magma", suficiente para inspeção visual sem
    # depender de matplotlib.
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
    rgb[mask] = 0.55 * rgb[mask] + 0.45 * np.array([0, 255, 255])
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    scale = 8
    thermal_image = Image.fromarray(rgb, mode="RGB").resize(
        (matrix.shape[1] * scale, matrix.shape[0] * scale),
        resample=Image.Resampling.NEAREST,
    )
    title_height = 56
    canvas = Image.new(
        "RGB",
        (thermal_image.width, thermal_image.height + title_height),
        "white",
    )
    canvas.paste(thermal_image, (0, title_height))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.multiline_text((8, 7), title, fill="black", font=font, spacing=3)

    top = title_height
    draw.rectangle(
        (
            x0 * scale,
            top + y0 * scale,
            x1 * scale - 1,
            top + y1 * scale - 1,
        ),
        outline="white",
        width=2,
    )
    seed_center_x = seed_x * scale + scale // 2
    seed_center_y = top + seed_y * scale + scale // 2
    cross_radius = scale
    draw.line(
        (
            seed_center_x - cross_radius,
            seed_center_y,
            seed_center_x + cross_radius,
            seed_center_y,
        ),
        fill="white",
        width=2,
    )
    draw.line(
        (
            seed_center_x,
            seed_center_y - cross_radius,
            seed_center_x,
            seed_center_y + cross_radius,
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
    masks_dir = output_dir / "masks"
    reviews_dir = output_dir / "revisao"
    output_dir.mkdir(parents=True, exist_ok=True)

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

    for index, row in eligible.iterrows():
        try:
            matrix_path, lot_name, animal_name = locate_matrix(
                data_dir,
                row["Lote"],
                row["id"],
                row["Foto"],
            )
            if not matrix_path.is_file():
                raise FileNotFoundError(f"Matriz não encontrada: {matrix_path}")

            matrix = np.load(matrix_path, allow_pickle=False)
            seed_x = int(round(float(row["Coord_X"])))
            seed_y = int(round(float(row["Coord_Y"])))
            if seed_x == 0 and seed_y == 0:
                raise ValueError(
                    "POI (0,0) tratado como marcação ausente; requer revisão manual."
                )
            height, width = matrix.shape
            if (
                seed_x < args.max_radius
                or seed_x >= width - args.max_radius
                or seed_y < args.max_radius
                or seed_y >= height - args.max_radius
            ):
                raise ValueError(
                    f"POI ({seed_x},{seed_y}) próximo demais da borda para "
                    f"uma área de raio {args.max_radius}; requer revisão manual."
                )
            result = region_growing_from_seed(
                matrix,
                seed_y,
                seed_x,
                tolerance=args.tolerance,
                max_radius=args.max_radius,
                reference_radius=args.reference_radius,
                connectivity=args.connectivity,
            )
            expected_search_size = 2 * args.max_radius + 1
            search_clipped = (
                result.search_bounds[2] - result.search_bounds[0]
                < expected_search_size
                or result.search_bounds[3] - result.search_bounds[1]
                < expected_search_size
            )
            features = extract_temperature_features(matrix, result.mask)
            quality, area_fraction, touches_limit = mask_quality_flags(
                result.mask,
                result.search_bounds,
                min_pixels=args.min_pixels,
                large_fraction=args.large_fraction,
                search_clipped=search_clipped,
            )

            stem = photo_stem(row["Foto"])
            relative_mask = (
                Path("masks")
                / safe_path_component(lot_name)
                / safe_path_component(animal_name)
                / f"{stem}.npy"
            )
            mask_path = output_dir / relative_mask
            if args.save_masks:
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(mask_path, result.mask)

            ambient_temperature = parse_temperature(row.get("Temp. Ambiente"))
            feature_row = row.to_dict()
            feature_row.update(
                {
                    "source_csv_row": int(index) + 2,
                    "matrix_path": str(matrix_path.relative_to(data_dir)),
                    "roi_mask_path": (
                        str(relative_mask.as_posix()) if args.save_masks else ""
                    ),
                    "seed_x": seed_x,
                    "seed_y": seed_y,
                    "seed_temperature": float(matrix[seed_y, seed_x]),
                    "reference_temperature": result.reference_temperature,
                    "roi_tolerance_c": args.tolerance,
                    "roi_max_radius_px": args.max_radius,
                    "roi_reference_radius_px": args.reference_radius,
                    "roi_connectivity": args.connectivity,
                    "roi_search_x0": result.search_bounds[0],
                    "roi_search_y0": result.search_bounds[1],
                    "roi_search_x1": result.search_bounds[2],
                    "roi_search_y1": result.search_bounds[3],
                    "roi_area_fraction": area_fraction,
                    "roi_touches_search_limit": touches_limit,
                    "roi_search_clipped": search_clipped,
                    "roi_quality": quality,
                    **features,
                }
            )
            if ambient_temperature is not None:
                feature_row["ambient_temperature_c"] = ambient_temperature
                feature_row["roi_mean_minus_ambient"] = (
                    features["roi_temp_mean"] - ambient_temperature
                )
                feature_row["roi_p90_minus_ambient"] = (
                    features["roi_temp_p90"] - ambient_temperature
                )
                feature_row["roi_max_minus_ambient"] = (
                    features["roi_temp_max"] - ambient_temperature
                )
            feature_rows.append(feature_row)
        except Exception as error:
            errors.append(
                {
                    "source_csv_row": int(index) + 2,
                    "id": row.get("id"),
                    "Lote": row.get("Lote"),
                    "Foto": row.get("Foto"),
                    "erro": str(error),
                }
            )

    features_dataframe = pd.DataFrame(feature_rows)
    features_path = output_dir / "roi_features.csv"
    features_dataframe.to_csv(
        features_path,
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )

    errors_path = output_dir / "roi_erros.csv"
    pd.DataFrame(errors).to_csv(
        errors_path,
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )

    if args.review_count > 0 and args.save_masks and feature_rows:
        review_count = min(args.review_count, len(feature_rows))
        sampled_positions = set(
            np.linspace(
                0,
                len(feature_rows) - 1,
                num=review_count,
                dtype=int,
            ).tolist()
        )
        critical_positions = {
            position
            for position, row in enumerate(feature_rows)
            if "ROI_PEQUENA" in str(row["roi_quality"])
            or "POI_PROXIMO_BORDA" in str(row["roi_quality"])
        }
        for position in sorted(sampled_positions | critical_positions):
            feature_row = feature_rows[position]
            matrix_path = data_dir / str(feature_row["matrix_path"])
            mask_path = output_dir / str(feature_row["roi_mask_path"])
            review_name = (
                f"linha_{int(feature_row['source_csv_row']):04d}_"
                f"{safe_path_component(str(feature_row['Lote']))}_"
                f"{photo_stem(feature_row['Foto'])}.png"
            )
            save_review_overlay(
                matrix_path,
                mask_path,
                reviews_dir / review_name,
                seed_x=int(feature_row["seed_x"]),
                seed_y=int(feature_row["seed_y"]),
                search_bounds=(
                    int(feature_row["roi_search_x0"]),
                    int(feature_row["roi_search_y0"]),
                    int(feature_row["roi_search_x1"]),
                    int(feature_row["roi_search_y1"]),
                ),
                title=(
                    f"{feature_row['Lote']} | Ovelha {feature_row['id']} | "
                    f"{photo_stem(feature_row['Foto'])}\n"
                    f"ROI={feature_row['roi_pixels']} px | "
                    f"{feature_row['roi_quality']}"
                ),
            )

    print(f"Dados: {data_dir}")
    print(f"Planilha: {csv_path}")
    print(f"Linhas com POI: {len(eligible)}")
    print(f"ROIs processadas: {len(feature_rows)}")
    print(f"Erros: {len(errors)}")
    print(f"Atributos: {features_path}")
    if args.save_masks:
        print(f"Máscaras: {masks_dir}")
    if args.review_count > 0 and args.save_masks:
        print(f"Amostras para revisão: {reviews_dir}")

    if not features_dataframe.empty:
        quality_counts = features_dataframe["roi_quality"].value_counts()
        print("Qualidade:")
        for quality, count in quality_counts.items():
            print(f"  {quality}: {count}")
        print(
            "Área da ROI (pixels): "
            f"mín={int(features_dataframe['roi_pixels'].min())}, "
            f"mediana={features_dataframe['roi_pixels'].median():.1f}, "
            f"máx={int(features_dataframe['roi_pixels'].max())}"
        )

    return 0 if feature_rows else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extrai ROIs térmicas por crescimento de região a partir de "
            "Coord_X/Coord_Y."
        )
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.3,
        help="Diferença térmica máxima em °C em relação à mediana local.",
    )
    parser.add_argument(
        "--max-radius",
        type=int,
        default=5,
        help="Raio espacial máximo; 5 corresponde a uma janela de 11x11.",
    )
    parser.add_argument(
        "--reference-radius",
        type=int,
        default=1,
        help="Raio da janela usada para calcular a mediana local; 1 = 3x3.",
    )
    parser.add_argument("--connectivity", type=int, choices=(4, 8), default=8)
    parser.add_argument("--min-pixels", type=int, default=3)
    parser.add_argument("--large-fraction", type=float, default=0.7)
    parser.add_argument("--review-count", type=int, default=24)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--save-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return process_dataset(args)
    except Exception as error:
        print(f"Erro: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
