"""Reconstrói matrizes radiométricas ausentes sem sobrescrever dados existentes.

O script cruza as fotografias referenciadas na planilha com a árvore processada
de dados. Para cada ``.npy`` ausente, procura o JPEG radiométrico primeiro no
destino esperado e depois na pasta bruta ``Dados/``.

Por segurança, a execução padrão apenas audita. Use ``--apply`` para copiar
JPEGs faltantes e criar as matrizes. Nenhum ``.npy`` existente é sobrescrito.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

from extrair_roi import (
    DEFAULT_CSV,
    PROJECT_ROOT,
    locate_matrix,
    photo_stem,
    resolve_data_dir,
)


DEFAULT_RAW_DIR = PROJECT_ROOT / "Dados"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "data_recovery"


@dataclass(frozen=True)
class RecoveryCandidate:
    source_csv_row: int
    animal_id: object
    lot: object
    photo: object
    target_matrix: Path
    target_jpeg: Path
    raw_source: Path | None
    source_status: str


def build_raw_index(raw_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    if not raw_dir.is_dir():
        return index

    for path in raw_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}:
            index.setdefault(path.stem.upper(), []).append(path.resolve())
    return index


def select_raw_source(
    stem: str,
    target_jpeg: Path,
    raw_index: dict[str, list[Path]],
) -> tuple[Path | None, str]:
    if target_jpeg.is_file():
        return target_jpeg, "jpeg_no_destino"

    matches = raw_index.get(stem.upper(), [])
    if len(matches) == 1:
        return matches[0], "jpeg_bruto_unico"
    if len(matches) > 1:
        return None, f"jpeg_bruto_ambiguo:{len(matches)}"
    return None, "jpeg_nao_encontrado"


def collect_candidates(
    dataframe: pd.DataFrame,
    data_dir: Path,
    raw_index: dict[str, list[Path]],
) -> tuple[list[RecoveryCandidate], int]:
    candidates: list[RecoveryCandidate] = []
    existing_matrices = 0

    for index, row in dataframe[dataframe["Foto"].notna()].iterrows():
        matrix_path, _, _ = locate_matrix(
            data_dir,
            row["Lote"],
            row["id"],
            row["Foto"],
        )
        if matrix_path.is_file():
            existing_matrices += 1
            continue

        target_jpeg = matrix_path.with_suffix(".jpg")
        stem = photo_stem(row["Foto"])
        raw_source, source_status = select_raw_source(
            stem,
            target_jpeg,
            raw_index,
        )
        candidates.append(
            RecoveryCandidate(
                source_csv_row=int(index) + 2,
                animal_id=row["id"],
                lot=row["Lote"],
                photo=row["Foto"],
                target_matrix=matrix_path,
                target_jpeg=target_jpeg,
                raw_source=raw_source,
                source_status=source_status,
            )
        )

    return candidates, existing_matrices


def extract_matrix_from_ascii_copy(source_jpeg: Path) -> np.ndarray:
    """Contorna a saída CP1252 do ExifTool quando o caminho contém acentos."""

    try:
        import flirimageextractor
    except ImportError as error:
        raise RuntimeError(
            "flirimageextractor não instalado. Execute "
            "'pip install -r requirements-modeling.txt'."
        ) from error

    with tempfile.TemporaryDirectory(prefix="flir_recovery_") as temp_dir:
        temporary_jpeg = Path(temp_dir) / source_jpeg.name
        shutil.copy2(source_jpeg, temporary_jpeg)
        extractor = flirimageextractor.FlirImageExtractor()
        extractor.process_image(str(temporary_jpeg))
        matrix = np.asarray(extractor.get_thermal_np(), dtype=np.float32)

    if matrix.shape != (60, 80):
        raise ValueError(
            f"Matriz reconstruída com formato {matrix.shape}; esperado (60, 80)."
        )
    if not np.isfinite(matrix).all():
        raise ValueError("Matriz reconstruída contém valores não finitos.")
    return matrix


def write_report(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_csv_row",
        "id",
        "Lote",
        "Foto",
        "source_status",
        "source_jpeg",
        "target_jpeg",
        "target_matrix",
        "result",
        "detail",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def process(args: argparse.Namespace) -> int:
    csv_path = args.csv.expanduser().resolve()
    data_dir = resolve_data_dir(args.data_dir)
    raw_dir = args.raw_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    dataframe = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
    raw_index = build_raw_index(raw_dir)
    candidates, existing_matrices = collect_candidates(
        dataframe,
        data_dir,
        raw_index,
    )

    report_rows: list[dict[str, object]] = []
    created = 0
    unavailable = 0
    errors = 0

    for candidate in candidates:
        result = "audit_only"
        detail = ""
        if candidate.raw_source is None:
            result = "unavailable"
            detail = candidate.source_status
            unavailable += 1
        elif args.apply:
            try:
                matrix = extract_matrix_from_ascii_copy(candidate.raw_source)
                candidate.target_matrix.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                if not candidate.target_jpeg.exists():
                    shutil.copy2(candidate.raw_source, candidate.target_jpeg)
                if candidate.target_matrix.exists():
                    result = "skipped_existing"
                    detail = "A matriz surgiu durante a execução; não sobrescrita."
                else:
                    np.save(candidate.target_matrix, matrix)
                    result = "created"
                    detail = (
                        f"shape={matrix.shape}; min={matrix.min():.3f}; "
                        f"max={matrix.max():.3f}"
                    )
                    created += 1
            except Exception as error:
                result = "error"
                detail = str(error)
                errors += 1

        report_rows.append(
            {
                "source_csv_row": candidate.source_csv_row,
                "id": candidate.animal_id,
                "Lote": candidate.lot,
                "Foto": photo_stem(candidate.photo),
                "source_status": candidate.source_status,
                "source_jpeg": (
                    str(candidate.raw_source)
                    if candidate.raw_source is not None
                    else ""
                ),
                "target_jpeg": str(candidate.target_jpeg),
                "target_matrix": str(candidate.target_matrix),
                "result": result,
                "detail": detail,
            }
        )

    report_path = output_dir / "matrix_recovery_report.csv"
    write_report(report_rows, report_path)

    mode = "APLICAÇÃO" if args.apply else "AUDITORIA"
    print(f"Modo: {mode}")
    print(f"Planilha: {csv_path}")
    print(f"Dados processados: {data_dir}")
    print(f"Dados brutos: {raw_dir}")
    print(f"Matrizes já existentes: {existing_matrices}")
    print(f"Matrizes ausentes encontradas: {len(candidates)}")
    print(f"Matrizes criadas: {created}")
    print(f"JPEGs indisponíveis/ambíguos: {unavailable}")
    print(f"Erros de extração: {errors}")
    print(f"Relatório: {report_path}")
    return 1 if errors else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audita e reconstrói matrizes radiométricas ausentes sem "
            "sobrescrever arquivos existentes."
        )
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Cria JPEGs/matrizes ausentes; sem esta opção, apenas audita.",
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
