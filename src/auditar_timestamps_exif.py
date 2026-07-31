"""Audita data, hora e proveniência dos JPEGs radiométricos FLIR.

O relatório preserva a data original da planilha, cria uma data analítica com
ano configurável e a compara com o campo EXIF ``DateTime`` do JPEG. Nenhum
arquivo de dados ou valor da planilha é alterado.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import ExifTags, Image, UnidentifiedImageError

from extrair_roi import (
    DEFAULT_CSV,
    PROJECT_ROOT,
    locate_matrix,
    photo_stem,
    resolve_data_dir,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "exif_timestamp_audit"
DATETIME_FORMATS = (
    "%Y:%m:%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y:%m:%d %H:%M",
    "%Y-%m-%d %H:%M",
)


def parse_exif_datetime(value: object) -> datetime | None:
    """Converte os formatos de data/hora mais comuns em EXIF."""

    if value is None:
        return None
    text = str(value).strip().replace("\x00", "")
    if not text:
        return None
    for date_format in DATETIME_FORMATS:
        try:
            return datetime.strptime(text, date_format)
        except ValueError:
            continue
    return None


def normalize_sheet_date(value: object, collection_year: int) -> pd.Timestamp:
    """Preserva dia/mês e troca apenas o ano para a análise."""

    parsed = pd.to_datetime(
        str(value).strip(),
        format="%d.%m.%Y",
        errors="raise",
    )
    return pd.Timestamp(
        year=collection_year,
        month=int(parsed.month),
        day=int(parsed.day),
    )


def build_jpeg_index(data_dir: Path) -> dict[str, list[Path]]:
    """Indexa JPEGs por nome, sem assumir capitalização da extensão."""

    index: dict[str, list[Path]] = {}
    for path in data_dir.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in {".jpg", ".jpeg"}
            and path.stem.upper().startswith("FLIR")
        ):
            index.setdefault(path.stem.upper(), []).append(path.resolve())
    return index


def resolve_jpeg(
    expected_path: Path,
    stem: str,
    jpeg_index: dict[str, list[Path]],
) -> tuple[Path | None, str, int]:
    """Prioriza o JPEG no destino esperado e evita escolha ambígua."""

    for extension in (".jpg", ".JPG", ".jpeg", ".JPEG"):
        candidate = expected_path.with_suffix(extension)
        if candidate.is_file():
            return candidate.resolve(), "expected_path", 1

    matches = jpeg_index.get(stem.upper(), [])
    if len(matches) == 1:
        return matches[0], "global_unique", 1
    if len(matches) > 1:
        return None, "ambiguous", len(matches)
    return None, "missing", 0


def read_exif(path: Path) -> dict[str, object]:
    """Lê somente metadados e dimensões, sem decodificar a matriz térmica."""

    with Image.open(path) as image:
        exif = {
            ExifTags.TAGS.get(tag, tag): value
            for tag, value in image.getexif().items()
        }
        return {
            "exif_datetime_raw": exif.get("DateTimeOriginal")
            or exif.get("DateTimeDigitized")
            or exif.get("DateTime"),
            "camera_make": exif.get("Make", ""),
            "camera_model": exif.get("Model", ""),
            "camera_software": exif.get("Software", ""),
            "image_width": int(image.width),
            "image_height": int(image.height),
            "image_mode": image.mode,
        }


def classify_timestamp(
    jpeg_status: str,
    exif_datetime: datetime | None,
    sheet_date: pd.Timestamp,
) -> str:
    if jpeg_status in {"missing", "ambiguous"}:
        return f"jpeg_{jpeg_status}"
    if exif_datetime is None:
        return "exif_datetime_missing_or_invalid"
    if exif_datetime.date() == sheet_date.date():
        return "date_match"
    return "date_mismatch"


def audit_records(
    dataframe: pd.DataFrame,
    *,
    data_dir: Path,
    collection_year: int,
) -> pd.DataFrame:
    jpeg_index = build_jpeg_index(data_dir)
    records: list[dict[str, object]] = []

    rows_with_photo = dataframe[dataframe["Foto"].notna()]
    for index, row in rows_with_photo.iterrows():
        stem = photo_stem(row["Foto"])
        matrix_path, lot_name, animal_name = locate_matrix(
            data_dir,
            row["Lote"],
            row["id"],
            row["Foto"],
        )
        jpeg_path, jpeg_status, candidate_count = resolve_jpeg(
            matrix_path.with_suffix(".jpg"),
            stem,
            jpeg_index,
        )
        sheet_date = normalize_sheet_date(row["Data"], collection_year)
        metadata: dict[str, object] = {
            "exif_datetime_raw": "",
            "camera_make": "",
            "camera_model": "",
            "camera_software": "",
            "image_width": pd.NA,
            "image_height": pd.NA,
            "image_mode": "",
        }
        read_error = ""
        if jpeg_path is not None:
            try:
                metadata = read_exif(jpeg_path)
            except (OSError, ValueError, UnidentifiedImageError) as error:
                read_error = f"{type(error).__name__}: {error}"

        exif_datetime = parse_exif_datetime(
            metadata.get("exif_datetime_raw")
        )
        timestamp_status = classify_timestamp(
            jpeg_status,
            exif_datetime,
            sheet_date,
        )
        if read_error:
            timestamp_status = "jpeg_read_error"

        exif_timestamp = (
            pd.Timestamp(exif_datetime) if exif_datetime is not None else pd.NaT
        )
        day_delta = (
            int((exif_timestamp.normalize() - sheet_date).days)
            if not pd.isna(exif_timestamp)
            else pd.NA
        )

        records.append(
            {
                "source_csv_row": int(index) + 2,
                "id": row["id"],
                "Lote": row["Lote"],
                "lot_directory": lot_name,
                "animal_directory": animal_name,
                "Foto": stem,
                "sheet_date_original": str(row["Data"]).strip(),
                "sheet_date_analytic": sheet_date.date().isoformat(),
                "jpeg_status": jpeg_status,
                "jpeg_candidate_count": candidate_count,
                "jpeg_path": str(jpeg_path) if jpeg_path else "",
                "timestamp_status": timestamp_status,
                "exif_datetime_raw": metadata["exif_datetime_raw"] or "",
                "exif_datetime": (
                    exif_datetime.isoformat(sep=" ")
                    if exif_datetime is not None
                    else ""
                ),
                "exif_date": (
                    exif_datetime.date().isoformat()
                    if exif_datetime is not None
                    else ""
                ),
                "exif_time": (
                    exif_datetime.time().isoformat()
                    if exif_datetime is not None
                    else ""
                ),
                "exif_hour": (
                    exif_datetime.hour if exif_datetime is not None else pd.NA
                ),
                "date_delta_days": day_delta,
                "camera_make": metadata["camera_make"],
                "camera_model": metadata["camera_model"],
                "camera_software": metadata["camera_software"],
                "image_width": metadata["image_width"],
                "image_height": metadata["image_height"],
                "image_mode": metadata["image_mode"],
                "jpeg_read_error": read_error,
            }
        )

    return pd.DataFrame.from_records(records)


def summarize_records(records: pd.DataFrame) -> pd.DataFrame:
    counts = Counter(records["timestamp_status"].astype(str))
    available = records["exif_datetime"].astype(str).str.len().gt(0)
    matched = records["timestamp_status"].eq("date_match")
    values: list[tuple[str, object]] = [
        ("records_with_photo", len(records)),
        ("jpeg_resolved", int(records["jpeg_path"].astype(bool).sum())),
        ("jpeg_missing", int((records["jpeg_status"] == "missing").sum())),
        (
            "jpeg_ambiguous",
            int((records["jpeg_status"] == "ambiguous").sum()),
        ),
        ("exif_datetime_available", int(available.sum())),
        ("analytic_date_matches_exif", int(matched.sum())),
        (
            "analytic_date_match_rate_among_exif",
            float(matched.sum() / available.sum()) if available.any() else 0.0,
        ),
        (
            "distinct_exif_dates",
            int(records.loc[available, "exif_date"].nunique()),
        ),
        (
            "minimum_exif_datetime",
            records.loc[available, "exif_datetime"].min()
            if available.any()
            else "",
        ),
        (
            "maximum_exif_datetime",
            records.loc[available, "exif_datetime"].max()
            if available.any()
            else "",
        ),
    ]
    values.extend(
        (f"status_{status}", count)
        for status, count in sorted(counts.items())
    )
    return pd.DataFrame(values, columns=["metric", "value"])


def count_by(
    records: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    return (
        records.groupby(list(columns), dropna=False)
        .size()
        .rename("records")
        .reset_index()
        .sort_values(list(columns))
    )


def run(args: argparse.Namespace) -> int:
    csv_path = args.csv.expanduser().resolve()
    data_dir = resolve_data_dir(args.data_dir)
    output_dir = args.output_dir.expanduser().resolve()
    dataframe = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")

    records = audit_records(
        dataframe,
        data_dir=data_dir,
        collection_year=args.collection_year,
    )
    summary = summarize_records(records)
    by_hour = count_by(records, ["exif_hour", "timestamp_status"])
    by_date = count_by(
        records,
        ["sheet_date_analytic", "exif_date", "timestamp_status"],
    )
    by_camera = count_by(
        records,
        ["camera_make", "camera_model", "image_width", "image_height"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = {
        "Registros": output_dir / "timestamp_audit_records.csv",
        "Resumo": output_dir / "timestamp_audit_summary.csv",
        "Por hora": output_dir / "timestamp_audit_by_hour.csv",
        "Por data": output_dir / "timestamp_audit_by_date.csv",
        "Por câmera": output_dir / "timestamp_audit_by_camera.csv",
    }
    records.to_csv(
        output_files["Registros"],
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        output_files["Resumo"],
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )
    by_hour.to_csv(
        output_files["Por hora"],
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )
    by_date.to_csv(
        output_files["Por data"],
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )
    by_camera.to_csv(
        output_files["Por câmera"],
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Planilha: {csv_path}")
    print(f"Dados: {data_dir}")
    print(f"Ano analítico: {args.collection_year}")
    for metric, value in summary.itertuples(index=False, name=None):
        print(f"{metric}: {value}")
    for label, path in output_files.items():
        print(f"{label}: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cruza datas da planilha com data/hora EXIF dos JPEGs FLIR sem "
            "alterar os dados originais."
        )
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--collection-year", type=int, default=2025)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
