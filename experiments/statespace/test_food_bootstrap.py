"""Checks for paired crossed-cluster resampling and template-family assignment."""
import unittest
import numpy as np

from food_bootstrap import comparisons, template_cluster, two_way_bootstrap


class BootstrapTests(unittest.TestCase):
    def test_pairing_preserves_constant_model_difference(self):
        x = np.array([.1, .4, .7, .8])
        values = np.stack([x, x + .1], axis=1)[:, :, None]
        draws, _ = two_way_bootstrap(values, ['a', 'a', 'b', 'b'],
                                     ['x', 'y', 'x', 'y'], n_boot=100)
        np.testing.assert_allclose(draws[:, 1, 0] - draws[:, 0, 0], .1, atol=1e-14)
        result = comparisons(values.mean(0), draws, ['a', 'b'])['a minus b']
        np.testing.assert_allclose(result['ci95'], [-.1, -.1], atol=1e-14)

    def test_product_weights_match_manual_first_resample(self):
        values = np.arange(4, dtype=float)[:, None, None]
        draws, _ = two_way_bootstrap(values, ['a', 'a', 'b', 'b'],
                                     ['x', 'y', 'x', 'y'], n_boot=1, seed=7)
        rng = np.random.default_rng(7)
        fw = np.bincount(rng.integers(2, size=2), minlength=2)
        tw = np.bincount(rng.integers(2, size=2), minlength=2)
        weights = np.outer(fw, tw).ravel()
        self.assertAlmostEqual(draws[0, 0, 0], np.average(np.arange(4), weights=weights))

    def test_one_cluster_on_both_axes_has_no_variation(self):
        values = np.arange(8, dtype=float).reshape(4, 2, 1)
        draws, empty = two_way_bootstrap(values, ['a'] * 4, ['x'] * 4, n_boot=20)
        np.testing.assert_allclose(draws, np.broadcast_to(values.mean(0), draws.shape))
        self.assertEqual(empty, 0)

    def test_sparse_intersections_are_redrawn_and_reproducible(self):
        values = np.array([0., 1.])[:, None, None]
        args = (values, ['a', 'b'], ['x', 'y'])
        first, empty = two_way_bootstrap(*args, n_boot=200)
        second, empty_again = two_way_bootstrap(*args, n_boot=200)
        self.assertGreater(empty, 0)
        self.assertEqual(empty, empty_again)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.isfinite(first).all())

    def test_alternate_surface_forms_share_generating_family(self):
        self.assertEqual(template_cluster('a|in hand', {'template': 'in hand'}),
                         template_cluster('b|held in hand', {'template': 'held in hand'}))
        self.assertNotEqual(template_cluster('a|raw', {'template': 'raw'}),
                            template_cluster('a|in hand', {'template': 'in hand'}))

    def test_missing_template_is_explicit_singleton(self):
        self.assertEqual(template_cluster('a|whole', {}), 'unrecorded:a|whole')
        self.assertNotEqual(template_cluster('a|whole', {}), template_cluster('b|whole', {}))


if __name__ == '__main__':
    unittest.main()
