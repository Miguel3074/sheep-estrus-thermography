from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from extrair_roi import extract_temperature_features, region_growing_from_seed
from extrair_roi_multiescala import fixed_square_mask, validate_window_sizes
from recuperar_matrizes_pendentes import select_raw_source
from anotador import poi_issue
from modelar_estro import encode_monta, parse_locale_numeric


class RegionGrowingTests(unittest.TestCase):
    def test_extracts_connected_temperature_region(self) -> None:
        matrix = np.full((9, 9), 30.0, dtype=np.float32)
        matrix[3:6, 3:6] = 37.0

        result = region_growing_from_seed(
            matrix,
            4,
            4,
            tolerance=0.5,
            max_radius=4,
            reference_radius=1,
            connectivity=8,
        )

        self.assertEqual(int(result.mask.sum()), 9)
        np.testing.assert_array_equal(result.mask[3:6, 3:6], True)

    def test_limits_growth_to_configured_radius(self) -> None:
        matrix = np.full((20, 20), 37.0, dtype=np.float32)

        result = region_growing_from_seed(
            matrix,
            10,
            10,
            tolerance=0.1,
            max_radius=2,
            reference_radius=1,
            connectivity=8,
        )

        self.assertEqual(int(result.mask.sum()), 25)
        self.assertEqual(result.search_bounds, (8, 8, 13, 13))

    def test_rejects_seed_outside_matrix(self) -> None:
        matrix = np.full((5, 5), 37.0, dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "fora da matriz"):
            region_growing_from_seed(matrix, 10, 10)

    def test_extracts_expected_features(self) -> None:
        matrix = np.array([[35.0, 36.0], [37.0, 38.0]], dtype=np.float32)
        mask = np.array([[False, True], [True, True]])

        features = extract_temperature_features(matrix, mask)

        self.assertEqual(features["roi_pixels"], 3)
        self.assertAlmostEqual(features["roi_temp_mean"], 37.0)
        self.assertAlmostEqual(features["roi_temp_median"], 37.0)
        self.assertAlmostEqual(features["roi_temp_max"], 38.0)


class FixedWindowTests(unittest.TestCase):
    def test_builds_centered_odd_square(self) -> None:
        mask = fixed_square_mask((60, 80), 30, 40, 11)

        self.assertEqual(mask.shape, (60, 80))
        self.assertEqual(mask.dtype, np.bool_)
        self.assertEqual(int(mask.sum()), 121)
        self.assertTrue(mask[30, 40])
        self.assertTrue(mask[25, 35])
        self.assertTrue(mask[35, 45])
        self.assertFalse(mask[24, 35])

    def test_rejects_even_window(self) -> None:
        with self.assertRaisesRegex(ValueError, "ímpares"):
            validate_window_sizes((10,))

    def test_rejects_window_outside_matrix(self) -> None:
        with self.assertRaisesRegex(ValueError, "ultrapassa"):
            fixed_square_mask((60, 80), 2, 2, 11)


class MatrixRecoveryTests(unittest.TestCase):
    def test_prefers_existing_destination_jpeg(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "FLIR0001.jpg"
            target.touch()

            source, status = select_raw_source("FLIR0001", target, {})

            self.assertEqual(source, target)
            self.assertEqual(status, "jpeg_no_destino")

    def test_accepts_only_unambiguous_raw_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "destino" / "FLIR0001.jpg"
            first = root / "a" / "FLIR0001.jpg"
            second = root / "b" / "FLIR0001.jpg"

            source, status = select_raw_source(
                "FLIR0001",
                target,
                {"FLIR0001": [first]},
            )
            ambiguous_source, ambiguous_status = select_raw_source(
                "FLIR0001",
                target,
                {"FLIR0001": [first, second]},
            )

            self.assertEqual(source, first)
            self.assertEqual(status, "jpeg_bruto_unico")
            self.assertIsNone(ambiguous_source)
            self.assertEqual(ambiguous_status, "jpeg_bruto_ambiguo:2")


class AnnotationAuditTests(unittest.TestCase):
    def test_classifies_missing_sentinel_border_and_valid_pois(self) -> None:
        shape = (60, 80)

        self.assertEqual(
            poi_issue(pd.Series({"Coord_X": np.nan, "Coord_Y": np.nan}), shape),
            "coordenadas_ausentes",
        )
        self.assertEqual(
            poi_issue(pd.Series({"Coord_X": 0, "Coord_Y": 0}), shape),
            "sentinela_0_0",
        )
        self.assertEqual(
            poi_issue(pd.Series({"Coord_X": 40, "Coord_Y": 55}), shape),
            "fora_da_area_valida_11x11",
        )
        self.assertIsNone(
            poi_issue(pd.Series({"Coord_X": 40, "Coord_Y": 30}), shape)
        )


class ModelingPreparationTests(unittest.TestCase):
    def test_parses_locale_numeric_values(self) -> None:
        parsed = parse_locale_numeric(
            pd.Series(["36,5", " 37.25 ", "", None])
        )

        self.assertAlmostEqual(parsed.iloc[0], 36.5)
        self.assertAlmostEqual(parsed.iloc[1], 37.25)
        self.assertTrue(pd.isna(parsed.iloc[2]))
        self.assertTrue(pd.isna(parsed.iloc[3]))

    def test_encodes_true_and_blank_monta(self) -> None:
        encoded = encode_monta(pd.Series(["true", None, "", "false", "sim"]))

        self.assertEqual(encoded.tolist(), [1, 0, 0, 0, 1])

    def test_rejects_unknown_monta_label(self) -> None:
        with self.assertRaisesRegex(ValueError, "não reconhecidos"):
            encode_monta(pd.Series(["talvez"]))


if __name__ == "__main__":
    unittest.main()
