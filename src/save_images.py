"""Gera somente as visualizações PNG ausentes das matrizes térmicas."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from extrair_roi import resolve_data_dir


def process(args: argparse.Namespace) -> int:
    import matplotlib.pyplot as plt

    data_dir = resolve_data_dir(args.data_dir)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else data_dir / "Termogramas_Gerados"
    )
    matrices = sorted(
        path
        for path in data_dir.rglob("*.npy")
        if output_dir not in path.parents
    )
    created = 0
    skipped = 0
    errors: list[tuple[Path, str]] = []

    for matrix_path in matrices:
        relative_path = matrix_path.relative_to(data_dir)
        output_path = output_dir / relative_path.with_suffix(".png")
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            matrix = np.load(matrix_path, allow_pickle=False)
            if matrix.shape != (60, 80):
                raise ValueError(f"formato inesperado: {matrix.shape}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.imsave(output_path, matrix, cmap="magma")
            created += 1
        except Exception as error:
            errors.append((matrix_path, str(error)))

    print(f"Dados: {data_dir}")
    print(f"Destino: {output_dir}")
    print(f"Matrizes encontradas: {len(matrices)}")
    print(f"PNGs criados: {created}")
    print(f"PNGs já existentes: {skipped}")
    print(f"Erros: {len(errors)}")
    for path, detail in errors[:10]:
        print(f"  - {path}: {detail}")
    return 1 if errors else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gera visualizações PNG ausentes a partir dos arquivos NPY."
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenera também PNGs já existentes.",
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
