from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from avaliar_alerta_personalizado import (  # noqa: E402
    add_personalized_features,
    calculate_event_metrics,
    event_starts,
    expected_top_k_counts,
    leave_one_out_group_median,
)


class HerdRelativeTests(unittest.TestCase):
    def test_median_excludes_current_animal(self) -> None:
        values = pd.Series([1.0, 2.0, 100.0, 8.0])
        groups = pd.Series(["day1", "day1", "day1", "day2"])
        result = leave_one_out_group_median(values, groups)
        self.assertEqual(result.iloc[0], 51.0)
        self.assertEqual(result.iloc[1], 50.5)
        self.assertEqual(result.iloc[2], 1.5)
        self.assertTrue(np.isnan(result.iloc[3]))


class PersonalizedFeatureTests(unittest.TestCase):
    @staticmethod
    def frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "animal_group": ["A", "A", "B", "B"],
                "collection_datetime": pd.to_datetime(
                    [
                        "2025-02-10",
                        "2025-02-11",
                        "2025-02-10",
                        "2025-02-11",
                    ]
                ),
                "collection_date": [
                    "2025-02-10",
                    "2025-02-11",
                    "2025-02-10",
                    "2025-02-11",
                ],
                "source_csv_row": [1, 2, 3, 4],
                "fixed_15_temp_mean": [30.0, 31.0, 32.0, 33.0],
                "fixed_15_temp_p90": [31.0, 32.0, 33.0, 34.0],
                "fixed_15_temp_max": [32.0, 33.0, 34.0, 35.0],
                "poi_temperature": [30.5, 31.5, 32.5, 33.5],
                "ambient_temperature_c": [25.0, 26.0, 25.0, 26.0],
            }
        )

    def test_future_row_does_not_change_past_features(self) -> None:
        original = self.frame()
        before = add_personalized_features(original, history_window=3)
        future = original.iloc[[0]].copy()
        future["collection_datetime"] = pd.Timestamp("2025-02-12")
        future["collection_date"] = "2025-02-12"
        future["source_csv_row"] = 5
        future["fixed_15_temp_mean"] = 99.0
        extended = add_personalized_features(
            pd.concat([original, future], ignore_index=True),
            history_window=3,
        )
        columns = [
            column
            for column in before
            if "personal_median" in column or "delta_" in column
        ]
        pd.testing.assert_frame_equal(
            before[columns],
            extended.iloc[: len(before)][columns],
        )

    def test_double_difference_uses_only_previous_history(self) -> None:
        result = add_personalized_features(
            self.frame(),
            history_window=3,
        )
        column = (
            "fixed_15_temp_mean_minus_herd_median"
            "_minus_personal_median"
        )
        first_a = result.loc[result["source_csv_row"].eq(1), column].iloc[0]
        second_a = result.loc[result["source_csv_row"].eq(2), column].iloc[0]
        self.assertTrue(np.isnan(first_a))
        self.assertEqual(second_a, 0.0)


class EventMetricTests(unittest.TestCase):
    def test_event_definition_and_one_to_one_matching(self) -> None:
        frame = pd.DataFrame(
            {
                "animal_group": ["A"] * 6,
                "collection_datetime": pd.date_range(
                    "2025-02-10",
                    periods=6,
                    freq="D",
                ),
                "target_mount_today": [0, 1, 1, 0, 1, 0],
                "prediction": [1, 1, 0, 1, 0, 0],
            }
        )
        self.assertEqual(
            len(event_starts(frame, merge_positive_gap_days=1)),
            2,
        )
        self.assertEqual(
            len(event_starts(frame, merge_positive_gap_days=2)),
            1,
        )
        metrics = calculate_event_metrics(
            frame,
            merge_positive_gap_days=1,
        )
        self.assertEqual(metrics["event_count"], 2)
        self.assertEqual(metrics["detected_events"], 2)
        self.assertEqual(metrics["alert_episodes"], 2)
        self.assertEqual(metrics["alert_precision"], 1.0)


class RankingTests(unittest.TestCase):
    def test_top_k_splits_tie_at_cutoff(self) -> None:
        true_positive, selected = expected_top_k_counts(
            np.array([1, 0, 1, 0]),
            np.array([0.9, 0.5, 0.5, 0.1]),
            top_k=2,
        )
        self.assertEqual(selected, 2)
        self.assertEqual(true_positive, 1.5)

    def test_all_tied_matches_prevalence_in_expectation(self) -> None:
        true_positive, selected = expected_top_k_counts(
            np.array([1, 0, 0, 0]),
            np.ones(4),
            top_k=2,
        )
        self.assertEqual(selected, 2)
        self.assertEqual(true_positive, 0.5)


if __name__ == "__main__":
    unittest.main()
