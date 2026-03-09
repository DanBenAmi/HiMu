"""Tests for fuzzy logic operators and tree evaluation."""

import numpy as np
import pytest

from himu.logic import (
    LogicEngine,
    fuzzy_and_product,
    fuzzy_and_min,
    fuzzy_and_geometric_mean,
    fuzzy_or,
    temporal_seq,
    fuzzy_right_after,
)


@pytest.fixture
def engine():
    return LogicEngine(kappa=2.0, fps=1.0, and_mode="product")


class TestFuzzyOperators:
    def test_and_product(self):
        a = np.array([0.8, 0.5, 0.2])
        b = np.array([0.6, 0.4, 0.9])
        result = fuzzy_and_product(a, b)
        np.testing.assert_allclose(result, a * b)

    def test_and_min(self):
        a = np.array([0.8, 0.5, 0.2])
        b = np.array([0.6, 0.4, 0.9])
        result = fuzzy_and_min(a, b)
        np.testing.assert_allclose(result, np.minimum(a, b))

    def test_and_geometric_mean(self):
        a = np.array([0.8, 0.5, 0.2])
        b = np.array([0.6, 0.4, 0.9])
        result = fuzzy_and_geometric_mean([a, b])
        expected = np.sqrt(a * b)
        np.testing.assert_allclose(result, expected)

    def test_or(self):
        a = np.array([0.8, 0.5, 0.0])
        b = np.array([0.6, 0.4, 1.0])
        result = fuzzy_or(a, b)
        expected = a + b - a * b
        np.testing.assert_allclose(result, expected)

    def test_or_identity(self):
        a = np.array([0.0, 0.0])
        b = np.array([0.0, 0.0])
        result = fuzzy_or(a, b)
        np.testing.assert_allclose(result, np.zeros(2))

    def test_seq_two_children(self):
        a = np.array([1.0, 0.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0, 1.0])
        result = temporal_seq([a, b])
        # Symmetric SEQ: both cause and effect get high scores
        assert result[0] > 0.0  # cause peak
        assert result[3] > 0.0  # effect peak
        # Middle frames (no signal) should be lower
        assert result[1] < result[0]
        assert result[2] < result[3]

    def test_right_after(self):
        cause = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
        effect = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        result = fuzzy_right_after(cause, effect, kappa=2.0, fps=1.0)
        # Effect right after cause should have non-zero score
        assert result[2] > 0.0
        # Score should decay with distance
        assert result[2] > result[4]


class TestTreeEvaluation:
    def test_leaf_node(self, engine):
        tree = {"op": "LEAF", "expert": "CLIP", "query": "sunset"}
        scores = {0: np.array([0.1, 0.9, 0.3])}
        result, _ = engine.evaluate_tree(tree, scores)
        np.testing.assert_allclose(result, scores[0])

    def test_and_tree(self, engine):
        tree = {
            "op": "AND",
            "children": [
                {"op": "LEAF", "expert": "CLIP", "query": "sunset"},
                {"op": "LEAF", "expert": "YOLO", "query": "car"},
            ],
        }
        scores = {
            0: np.array([0.8, 0.2]),
            1: np.array([0.6, 0.9]),
        }
        result, _ = engine.evaluate_tree(tree, scores)
        expected = scores[0] * scores[1]
        np.testing.assert_allclose(result, expected)

    def test_or_tree(self, engine):
        tree = {
            "op": "OR",
            "children": [
                {"op": "LEAF", "expert": "CLIP", "query": "a"},
                {"op": "LEAF", "expert": "CLIP", "query": "b"},
            ],
        }
        a = np.array([0.8, 0.0])
        b = np.array([0.0, 0.7])
        scores = {0: a, 1: b}
        result, _ = engine.evaluate_tree(tree, scores)
        expected = a + b - a * b
        np.testing.assert_allclose(result, expected)

    def test_rescaled_tree(self, engine):
        tree = {
            "op": "AND",
            "children": [
                {"op": "LEAF", "expert": "CLIP", "query": "a"},
                {"op": "LEAF", "expert": "CLIP", "query": "b"},
            ],
        }
        scores = {
            0: np.array([0.0, 1.0]),
            1: np.array([1.0, 0.0]),
        }
        result, _ = engine.evaluate_tree_rescaled(tree, scores, lo=0.5, hi=1.0)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_collect_leaf_nodes(self, engine):
        tree = {
            "op": "AND",
            "children": [
                {"op": "LEAF", "expert": "CLIP", "query": "sunset"},
                {"op": "OR",
                 "children": [
                     {"op": "LEAF", "expert": "YOLO", "query": "car"},
                     {"op": "LEAF", "expert": "ASR", "query": "driving"},
                 ]},
            ],
        }
        leaves = engine.collect_leaf_nodes(tree)
        assert len(leaves) == 3
        experts = {leaf["expert"] for _, leaf in leaves}
        assert experts == {"CLIP", "YOLO", "ASR"}

    def test_group_leaves_by_expert(self, engine):
        leaves = [
            (0, {"op": "LEAF", "expert": "CLIP", "query": "a"}),
            (1, {"op": "LEAF", "expert": "CLIP", "query": "b"}),
            (2, {"op": "LEAF", "expert": "YOLO", "query": "c"}),
        ]
        grouped = engine.group_leaves_by_expert(leaves)
        assert len(grouped["CLIP"]) == 2
        assert len(grouped["YOLO"]) == 1
