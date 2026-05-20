import tempfile
import unittest
from pathlib import Path

import numpy as np

from pyscf.mpcc import laplace_quadrature


class KnownLaplaceQuadratureTest(unittest.TestCase):
    def test_parse_laplace_minimax_output(self):
        text = """
     range or orbital energy denominator:    0.200E+01    0.800E+01
  maximum absolute error of distribution:    0.100E-08

                  exponents              weights
    =================================================
       1         0.1250000000         0.5000000000
       2         0.7500000000         1.2500000000

"""
        quad = laplace_quadrature.parse_laplace_minimax_output(text)

        self.assertEqual(quad.nlap, 2)
        self.assertAlmostEqual(quad.ymin, 2.0)
        self.assertAlmostEqual(quad.ymax, 8.0)
        self.assertAlmostEqual(quad.errmax, 1.0e-9)
        np.testing.assert_allclose(quad.exponents, [0.125, 0.75])
        np.testing.assert_allclose(quad.weights, [0.5, 1.25])

    def test_reference_interval_scaling(self):
        quad = laplace_quadrature.from_reference_interval(
            exponents=np.array([2.0, 4.0]),
            weights=np.array([3.0, 5.0]),
            ymin=2.0,
            ymax=20.0,
        )

        np.testing.assert_allclose(quad.exponents, [1.0, 2.0])
        np.testing.assert_allclose(quad.weights, [1.5, 2.5])

    def test_save_load_roundtrip(self):
        quad = laplace_quadrature.LaplaceQuadrature(
            np.array([0.1, 0.2]),
            np.array([0.3, 0.4]),
            1.0,
            10.0,
            1.0e-8,
            "unit-test",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = Path(tmpdir) / "quad.npz"
            quad.save(filename)
            loaded = laplace_quadrature.load(filename)

        np.testing.assert_allclose(loaded.exponents, quad.exponents)
        np.testing.assert_allclose(loaded.weights, quad.weights)
        self.assertEqual(loaded.ymin, quad.ymin)
        self.assertEqual(loaded.ymax, quad.ymax)
        self.assertEqual(loaded.errmax, quad.errmax)
        self.assertEqual(loaded.source, quad.source)


class LaplaceMinimaxInitTableTest(unittest.TestCase):
    def test_load_pretabulated_table(self):
        root = Path(__file__).resolve().parents[3] / "external" / "laplace-minimax"
        if not root.exists():
            self.skipTest("laplace-minimax submodule is not available")

        quad = laplace_quadrature.from_init_table(root, ymin=2.0, ymax=8.0, nlap=8)
        self.assertEqual(quad.nlap, 8)
        self.assertEqual(quad.ymin, 2.0)
        self.assertEqual(quad.ymax, 8.0)
        self.assertTrue(np.all(quad.exponents > 0.0))
        self.assertTrue(np.all(quad.weights > 0.0))

        stats = quad.validate(ngrid=1000)
        self.assertLess(stats["max_rel"], 1.0e-4)


if __name__ == "__main__":
    unittest.main()
