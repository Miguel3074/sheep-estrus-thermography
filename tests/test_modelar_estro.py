from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from modelar_estro import (
    add_temporal_features,
    make_splits,
    normalize_collection_dates,
)


class CollectionDateTests(unittest.TestCase):
    def test_normalizes_incremental_years_without_changing_day_month(self) -> None:
        normalized, corrected = normalize_collection_dates(
            pd.Series(["21.02.2025", "22.02.2027", "28.02.2033"]),
            collection_year=2025,
        )

        self.assertEqual(
            normalized.tolist(),
            ["2025-02-21", "2025-02-22", "2025-02-28"],
        )
        self.assertEqual(corrected.tolist(), [False, True, True])

    def test_rejects_invalid_date(self) -> None:
        with self.assertRaisesRegex(ValueError, "inválidas"):
            normalize_collection_dates(
                pd.Series(["31.02.2025"]),
                collection_year=2025,
            )


class GroupedValidationTests(unittest.TestCase):
    def test_date_groups_never_overlap(self) -> None:
        dataframe = pd.DataFrame(
            {
                "target_monta": np.tile([0, 1], 6),
                "animal_group": np.repeat(["a", "b", "c"], 4),
                "collection_date": np.repeat(
                    [f"2025-02-{day:02d}" for day in range(1, 7)],
                    2,
                ),
            }
        )

        splits = make_splits(
            dataframe,
            folds=3,
            repeats=2,
            seed=42,
            group_by="date",
        )

        self.assertEqual(len(splits), 6)
        dates = dataframe["collection_date"].to_numpy()
        for _, _, train_index, test_index in splits:
            self.assertFalse(
                set(dates[train_index]) & set(dates[test_index])
            )


class TemporalFeatureTests(unittest.TestCase):
    @staticmethod
    def sample_dataframe() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "animal_group": ["a", "a", "a", "b"],
                "collection_date": [
                    "2025-02-01",
                    "2025-02-02",
                    "2025-02-04",
                    "2025-02-01",
                ],
                "source_csv_row": [1, 2, 3, 4],
                "fixed_11_temp_mean": [10.0, 12.0, 18.0, 30.0],
                "fixed_11_temp_p90": [11.0, 13.0, 19.0, 31.0],
                "fixed_11_temp_max": [12.0, 14.0, 20.0, 32.0],
                "ambient_temperature_c": [20.0, 21.0, 23.0, 25.0],
            }
        )

    def test_uses_only_previous_measurements_from_same_animal(self) -> None:
        temporal = add_temporal_features(self.sample_dataframe())

        self.assertTrue(pd.isna(temporal.loc[0, "days_since_previous"]))
        self.assertEqual(temporal.loc[1, "days_since_previous"], 1.0)
        self.assertEqual(temporal.loc[2, "days_since_previous"], 2.0)
        self.assertTrue(pd.isna(temporal.loc[3, "days_since_previous"]))
        self.assertEqual(
            temporal.loc[2, "fixed_11_temp_mean_delta_previous"],
            6.0,
        )
        self.assertEqual(
            temporal.loc[2, "fixed_11_temp_mean_delta_per_day"],
            3.0,
        )
        self.assertEqual(
            temporal.loc[2, "fixed_11_temp_mean_minus_recent_median"],
            7.0,
        )

    def test_future_measurement_does_not_change_past_features(self) -> None:
        original = self.sample_dataframe()
        altered = original.copy()
        altered.loc[2, "fixed_11_temp_mean"] = 99.0

        before = add_temporal_features(original)
        after = add_temporal_features(altered)
        columns = [
            "fixed_11_temp_mean_delta_previous",
            "fixed_11_temp_mean_delta_per_day",
            "fixed_11_temp_mean_minus_recent_median",
        ]
        pd.testing.assert_frame_equal(
            before.loc[:1, columns],
            after.loc[:1, columns],
        )


if __name__ == "__main__":
    unittest.main()
