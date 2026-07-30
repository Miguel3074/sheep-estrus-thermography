"""Interface para revisar somente os POIs ausentes ou inválidos.

O clique é feito na matriz radiométrica 80 x 60. Antes da primeira alteração,
o CSV recebe uma cópia de segurança datada. Cada clique válido é persistido
atomicamente para permitir interromper e retomar o trabalho.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

from extrair_roi import DEFAULT_CSV, PROJECT_ROOT, locate_matrix, resolve_data_dir
from extrair_roi_multiescala import PRIMARY_WINDOW_SIZE


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "annotation"


def poi_issue(row: pd.Series, matrix_shape: tuple[int, int]) -> str | None:
    if pd.isna(row.get("Coord_X")) or pd.isna(row.get("Coord_Y")):
        return "coordenadas_ausentes"

    seed_x = int(round(float(row["Coord_X"])))
    seed_y = int(round(float(row["Coord_Y"])))
    if seed_x == 0 and seed_y == 0:
        return "sentinela_0_0"

    height, width = matrix_shape
    radius = PRIMARY_WINDOW_SIZE // 2
    if not (
        radius <= seed_x < width - radius
        and radius <= seed_y < height - radius
    ):
        return "fora_da_area_valida_11x11"
    return None


def collect_pending(
    dataframe: pd.DataFrame,
    data_dir: Path,
) -> list[dict[str, object]]:
    pending: list[dict[str, object]] = []
    for index, row in dataframe[dataframe["Foto"].notna()].iterrows():
        matrix_path, lot_name, animal_name = locate_matrix(
            data_dir,
            row["Lote"],
            row["id"],
            row["Foto"],
        )
        if matrix_path.is_file():
            matrix = np.load(matrix_path, allow_pickle=False, mmap_mode="r")
            shape = tuple(matrix.shape)
            issue = poi_issue(row, shape)
        else:
            shape = None
            issue = (
                "matriz_ausente"
                if pd.isna(row.get("Coord_X"))
                or pd.isna(row.get("Coord_Y"))
                else None
            )

        if issue is None:
            continue

        pending.append(
            {
                "dataframe_index": int(index),
                "source_csv_row": int(index) + 2,
                "id": row["id"],
                "Lote": row["Lote"],
                "Foto": row["Foto"],
                "Data": row.get("Data"),
                "Coord_X_atual": row.get("Coord_X"),
                "Coord_Y_atual": row.get("Coord_Y"),
                "issue": issue,
                "matrix_available": matrix_path.is_file(),
                "matrix_shape": str(shape) if shape is not None else "",
                "matrix_path": str(matrix_path),
                "lot_name": lot_name,
                "animal_name": animal_name,
            }
        )
    return pending


def save_dataframe_atomically(dataframe: pd.DataFrame, csv_path: Path) -> None:
    temporary_path = csv_path.with_name(f".{csv_path.name}.tmp")
    dataframe.to_csv(
        temporary_path,
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )
    temporary_path.replace(csv_path)


def make_backup(csv_path: Path) -> Path:
    backup_dir = csv_path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{csv_path.stem}_{timestamp}{csv_path.suffix}"
    shutil.copy2(csv_path, backup_path)
    return backup_path


def select_poi(
    matrix: np.ndarray,
    jpeg_path: Path,
    *,
    title: str,
    current_x: object,
    current_y: object,
) -> tuple[int, int] | None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    if jpeg_path.is_file():
        figure, (visual_axis, thermal_axis) = plt.subplots(
            1,
            2,
            figsize=(14, 6),
        )
        with Image.open(jpeg_path) as visual:
            visual_axis.imshow(visual.convert("RGB"))
        visual_axis.set_title("Referência visual — não clicar")
        visual_axis.axis("off")
    else:
        figure, thermal_axis = plt.subplots(1, 1, figsize=(8, 6))

    figure.canvas.manager.set_window_title(title)
    thermal_axis.imshow(matrix, cmap="magma", interpolation="nearest")
    thermal_axis.set_title(
        "Clique no centro da vulva — feche a janela para interromper"
    )
    thermal_axis.set_xlabel("x")
    thermal_axis.set_ylabel("y")

    radius = PRIMARY_WINDOW_SIZE // 2
    valid_width = matrix.shape[1] - 2 * radius
    valid_height = matrix.shape[0] - 2 * radius
    thermal_axis.add_patch(
        Rectangle(
            (radius - 0.5, radius - 0.5),
            valid_width,
            valid_height,
            fill=False,
            edgecolor="cyan",
            linewidth=1.5,
            linestyle="--",
        )
    )

    if pd.notna(current_x) and pd.notna(current_y):
        thermal_axis.plot(
            [float(current_x)],
            [float(current_y)],
            marker="x",
            color="red",
            markersize=10,
            markeredgewidth=2,
            label="POI anterior",
        )
        thermal_axis.legend(loc="upper right")

    plt.tight_layout()
    while True:
        points = plt.ginput(1, timeout=0)
        if not points:
            plt.close(figure)
            return None

        seed_x = int(round(points[0][0]))
        seed_y = int(round(points[0][1]))
        if (
            radius <= seed_x < matrix.shape[1] - radius
            and radius <= seed_y < matrix.shape[0] - radius
        ):
            plt.close(figure)
            return seed_x, seed_y

        print(
            f"POI ({seed_x},{seed_y}) fora da área ciano válida para "
            f"{PRIMARY_WINDOW_SIZE}x{PRIMARY_WINDOW_SIZE}; clique novamente."
        )


def process(args: argparse.Namespace) -> int:
    csv_path = args.csv.expanduser().resolve()
    data_dir = resolve_data_dir(args.data_dir)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
    pending = collect_pending(dataframe, data_dir)
    pending_dataframe = pd.DataFrame(pending)
    pending_path = output_dir / "pending_pois.csv"
    pending_dataframe.to_csv(
        pending_path,
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )

    actionable = [item for item in pending if item["matrix_available"]]
    unavailable = [item for item in pending if not item["matrix_available"]]
    print(f"POIs pendentes: {len(pending)}")
    print(f"Prontos para marcação: {len(actionable)}")
    print(f"Sem matriz: {len(unavailable)}")
    print(f"Lista: {pending_path}")

    if args.audit:
        return 0

    if args.limit is not None:
        actionable = actionable[: args.limit]
    if not actionable:
        print("Nenhum POI disponível para marcação.")
        return 0

    backup_path = make_backup(csv_path)
    print(f"Backup criado: {backup_path}")
    saved = 0

    for position, item in enumerate(actionable, start=1):
        matrix_path = Path(str(item["matrix_path"]))
        matrix = np.load(matrix_path, allow_pickle=False)
        jpeg_path = matrix_path.with_suffix(".jpg")
        title = (
            f"{position}/{len(actionable)} | {item['lot_name']} | "
            f"{item['animal_name']} | {matrix_path.stem} | {item['issue']}"
        )
        print(title)
        selected = select_poi(
            matrix,
            jpeg_path,
            title=title,
            current_x=item["Coord_X_atual"],
            current_y=item["Coord_Y_atual"],
        )
        if selected is None:
            print("Anotação interrompida pelo usuário.")
            break

        seed_x, seed_y = selected
        dataframe.at[item["dataframe_index"], "Coord_X"] = seed_x
        dataframe.at[item["dataframe_index"], "Coord_Y"] = seed_y
        save_dataframe_atomically(dataframe, csv_path)
        saved += 1
        print(f"Salvo: POI=({seed_x},{seed_y})")

    print(f"POIs salvos nesta sessão: {saved}")
    print("Execute novamente para retomar os casos restantes.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audita ou revisa POIs ausentes/inválidos com backup."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Somente gera a lista de pendências, sem abrir janelas.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limita a quantidade de imagens nesta sessão interativa.",
    )
    return parser


def main() -> int:
    try:
        return process(build_parser().parse_args())
    except Exception as error:
        print(f"Erro: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
