"""Tests for score normalization."""

import numpy as np
import pytest

from himu.normalization import ScoreNormalizer


class TestRobustNormalization:
    def test_output_range(self):
        normalizer = ScoreNormalizer(method="robust", gamma=3.0)
        scores = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
        result = normalizer.normalize(scores)
        assert np.all(result > 0.0)
        assert np.all(result < 1.0)

    def test_monotonicity(self):
        normalizer = ScoreNormalizer(method="robust", gamma=3.0)
        scores = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        result = normalizer.normalize(scores)
        assert np.all(np.diff(result) > 0)

    def test_constant_input(self):
        normalizer = ScoreNormalizer(method="robust", gamma=3.0)
        scores = np.full(10, 0.5)
        result = normalizer.normalize(scores)
        np.testing.assert_allclose(result, np.full(10, 0.5), atol=1e-5)


class TestMinmaxNormalization:
    def test_output_range(self):
        normalizer = ScoreNormalizer(method="minmax")
        scores = np.array([0.1, 0.3, 0.7, 0.9])
        result = normalizer.normalize(scores)
        assert np.isclose(result.min(), 0.0)
        assert np.isclose(result.max(), 1.0)

    def test_constant_input(self):
        normalizer = ScoreNormalizer(method="minmax")
        scores = np.full(5, 0.5)
        result = normalizer.normalize(scores)
        np.testing.assert_allclose(result, np.zeros(5))


class TestZscoreNormalization:
    def test_output_range(self):
        normalizer = ScoreNormalizer(method="zscore", gamma=3.0)
        scores = np.array([0.0, 0.5, 1.0])
        result = normalizer.normalize(scores)
        assert np.all(result > 0.0)
        assert np.all(result < 1.0)


class TestJointNormalization:
    def test_joint_robust(self):
        normalizer = ScoreNormalizer(method="robust", gamma=3.0)
        scores_dict = {
            "query_a": np.array([0.0, 0.5, 1.0]),
            "query_b": np.array([0.2, 0.3, 0.4]),
        }
        result = normalizer.normalize_joint(scores_dict)
        assert set(result.keys()) == {"query_a", "query_b"}
        assert result["query_a"].max() > result["query_b"].max()

    def test_single_signal_matches_independent(self):
        normalizer = ScoreNormalizer(method="robust", gamma=3.0)
        scores = np.array([0.1, 0.5, 0.9])
        independent = normalizer.normalize(scores)
        joint = normalizer.normalize_joint({"q": scores})
        np.testing.assert_allclose(joint["q"], independent)

    def test_joint_minmax(self):
        normalizer = ScoreNormalizer(method="minmax")
        scores_dict = {
            "a": np.array([0.0, 0.5]),
            "b": np.array([0.5, 1.0]),
        }
        result = normalizer.normalize_joint(scores_dict)
        assert np.isclose(result["a"].min(), 0.0)
        assert np.isclose(result["b"].max(), 1.0)
