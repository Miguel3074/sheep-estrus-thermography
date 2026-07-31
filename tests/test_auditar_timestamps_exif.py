from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from auditar_timestamps_exif import (  # noqa: E402
    classify_timestamp,
    normalize_sheet_date,
    parse_exif_datetime,
    resolve_jpeg,
)


class ExifTimestampParsingTests(unittest.TestCase):
    def test_parses_standard_exif_datetime(self) -> None:
        value = parse_exif_datetime("2025:02:13 19:04:20")

        self.assertEqual(value, datetime(2025, 2, 13, 19, 4, 20))

    def test_normalizes_only_collection_year(self) -> None:
        value = normalize_sheet_date("21.02.2031", 2025)

        self.assertEqual(value, pd.Timestamp("2025-02-21"))

    def test_classifies_date_match_and_mismatch(self) -> None:
        sheet_date = pd.Timestamp("2025-02-13")

        self.assertEqual(
            classify_timestamp(
                "expected_path",
                datetime(2025, 2, 13, 19, 4),
                sheet_date,
            ),
            "date_match",
        )
        self.assertEqual(
            classify_timestamp(
                "expected_path",
                datetime(2025, 2, 14, 7, 0),
                sheet_date,
            ),
            "date_mismatch",
        )


class JpegResolutionTests(unittest.TestCase):
    def test_prefers_expected_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            expected = Path(temp_dir) / "FLIR0001.jpg"
            expected.touch()

            path, status, count = resolve_jpeg(
                expected,
                "FLIR0001",
                {"FLIR0001": [Path(temp_dir) / "other.jpg"]},
            )

            self.assertEqual(path, expected.resolve())
            self.assertEqual(status, "expected_path")
            self.assertEqual(count, 1)

    def test_refuses_ambiguous_global_match(self) -> None:
        first = Path("a") / "FLIR0001.jpg"
        second = Path("b") / "FLIR0001.jpg"

        path, status, count = resolve_jpeg(
            Path("missing") / "FLIR0001.jpg",
            "FLIR0001",
            {"FLIR0001": [first, second]},
        )

        self.assertIsNone(path)
        self.assertEqual(status, "ambiguous")
        self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()
