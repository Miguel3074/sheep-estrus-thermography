from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from avaliar_janela_24h import (  # noqa: E402
    apply_exif_collection_dates,
    connected_temperature_band,
    make_timed_window_target,
    make_window_target,
    purge_adjacent_dates,
    select_f1_threshold,
)
from bootstrap_janela_24h import cluster_bootstrap  # noqa: E402


class WindowTargetTests(unittest.TestCase):
    def test_window_uses_current_and_next_calendar_day(self) -> None:
        frame = pd.DataFrame(
            {
                "animal_group": ["A", "A", "A", "B"],
                "collection_datetime": pd.to_datetime(
                    [
                        "2025-02-10",
                        "2025-02-11",
                        "2025-02-13",
                        "2025-02-11",
                    ]
                ),
            }
        )
        mount = pd.Series([0, 1, 0, 0], dtype="int8")
        result = make_window_target(frame, mount, future_days=1)
        self.assertEqual(result.tolist(), [1, 1, 0, 0])

    def test_can_build_strictly_next_day_target(self) -> None:
        frame = pd.DataFrame(
            {
                "animal_group": ["A", "A", "A"],
                "collection_datetime": pd.to_datetime(
                    ["2025-02-10", "2025-02-11", "2025-02-12"]
                ),
            }
        )
        mount = pd.Series([0, 1, 0], dtype="int8")

        result = make_window_target(
            frame,
            mount,
            start_offset=1,
            future_days=1,
        )

        self.assertEqual(result.tolist(), [1, 0, 0])

    def test_builds_exact_future_24_hour_target(self) -> None:
        frame = pd.DataFrame(
            {
                "animal_group": ["A", "A", "A", "A"],
                "collection_datetime": pd.to_datetime(
                    [
                        "2025-02-10 17:30:00",
                        "2025-02-10 19:00:00",
                        "2025-02-11 18:00:00",
                        "2025-02-12 17:00:00",
                    ]
                ),
            }
        )
        mount = pd.Series([0, 0, 1, 0], dtype="int8")

        result = make_timed_window_target(
            frame,
            mount,
            horizon_hours=24,
        )

        self.assertEqual(result.tolist(), [0, 1, 0, 0])

    def test_applies_verified_exif_date_without_losing_sheet_date(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audit_path = Path(temp_dir) / "audit.csv"
            pd.DataFrame(
                {
                    "source_csv_row": [2],
                    "exif_date": ["2025-03-01"],
                    "exif_datetime": ["2025-03-01 18:25:42"],
                    "timestamp_status": ["date_mismatch"],
                }
            ).to_csv(
                audit_path,
                sep=";",
                index=False,
                encoding="utf-8-sig",
            )
            frame = pd.DataFrame(
                {
                    "source_csv_row": [2],
                    "collection_datetime": pd.to_datetime(["2025-02-01"]),
                }
            )

            result = apply_exif_collection_dates(frame, audit_path)

            self.assertEqual(
                result.loc[0, "collection_datetime"],
                pd.Timestamp("2025-03-01 18:25:42"),
            )
            self.assertEqual(
                result.loc[0, "collection_datetime_sheet"],
                pd.Timestamp("2025-02-01"),
            )
            self.assertEqual(result.loc[0, "collection_date_source"], "exif")
            self.assertEqual(result.loc[0, "collection_date_exif_changed"], 1)

    def test_purge_removes_adjacent_training_dates(self) -> None:
        dates = pd.Series(
            pd.to_datetime(
                [
                    "2025-02-10",
                    "2025-02-11",
                    "2025-02-12",
                    "2025-02-14",
                ]
            )
        )
        result = purge_adjacent_dates(
            np.array([0, 1, 3]),
            np.array([2]),
            dates,
            purge_days=1,
        )
        self.assertEqual(result.tolist(), [0, 3])


class RegionTests(unittest.TestCase):
    def test_connected_band_respects_temperature_and_radius(self) -> None:
        matrix = np.full((9, 9), 30.0)
        matrix[3:6, 3:6] = 35.0
        matrix[4, 5] = 35.05
        matrix[4, 6] = 35.4
        mask, touches = connected_temperature_band(
            matrix,
            seed_x=4,
            seed_y=4,
            radius=2,
            tolerance_c=0.1,
        )
        self.assertTrue(mask[4, 4])
        self.assertTrue(mask[4, 5])
        self.assertFalse(mask[4, 6])
        self.assertFalse(touches)

    def test_threshold_returns_valid_f1(self) -> None:
        threshold, f1 = select_f1_threshold(
            np.array([0, 0, 1, 1]),
            np.array([0.1, 0.4, 0.6, 0.9]),
        )
        self.assertGreaterEqual(threshold, 0.4)
        self.assertLessEqual(threshold, 0.9)
        self.assertEqual(f1, 1.0)


class BootstrapTests(unittest.TestCase):
    def test_cluster_bootstrap_returns_paired_difference(self) -> None:
        rows = []
        for grouping in ("animal", "date"):
            for source_row, truth, group in (
                (1, 0, "A"),
                (2, 0, "A"),
                (3, 1, "B"),
                (4, 1, "B"),
            ):
                rows.extend(
                    [
                        {
                            "grouping": grouping,
                            "feature_set": "fixed_15",
                            "source_csv_row": source_row,
                            "truth": truth,
                            grouping: group,
                            "score": [0.1, 0.2, 0.8, 0.9][
                                source_row - 1
                            ],
                        },
                        {
                            "grouping": grouping,
                            "feature_set": "candidate",
                            "source_csv_row": source_row,
                            "truth": truth,
                            grouping: group,
                            "score": [0.2, 0.3, 0.7, 0.8][
                                source_row - 1
                            ],
                        },
                    ]
                )
        absolute, differences = cluster_bootstrap(
            pd.DataFrame(rows),
            reference="fixed_15",
            iterations=30,
            seed=7,
        )
        self.assertFalse(absolute.empty)
        self.assertFalse(differences.empty)
        self.assertTrue(
            differences["observed_difference"].notna().all()
        )


if __name__ == "__main__":
    unittest.main()
