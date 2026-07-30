from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from modelar_estro import make_splits, normalize_collection_dates


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


if __name__ == "__main__":
    unittest.main()
