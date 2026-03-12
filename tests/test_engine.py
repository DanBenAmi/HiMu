"""Tests for the pipeline integration (stages 3-5 with synthetic data)."""

import numpy as np
import pytest

from himu.logic import LogicEngine
from himu.normalization import ScoreNormalizer
from himu.smoothing import BandwidthMatchedSmoother, EXPERT_MODALITY_MAP
from himu.selection import create_selector


class TestPipelineIntegration:
    """End-to-end test of stages 3-5 with synthetic expert signals."""

    def test_stages_3_to_5(self):
        num_frames = 100
        fps = 1.0

        # Synthetic tree: AND(CLIP:sunset, OVD:car)
        tree = {
            "op": "AND",
            "children": [
                {"op": "LEAF", "expert": "CLIP", "query": "sunset"},
                {"op": "LEAF", "expert": "OVD", "query": "car"},
            ],
        }

        # Synthetic raw scores
        clip_raw = np.random.rand(num_frames).astype(np.float32)
        clip_raw[40:50] = 0.9  # sunset region

        ovd_raw = np.random.rand(num_frames).astype(np.float32) * 0.3
        ovd_raw[42:48] = 0.85  # car visible during sunset

        # Stage 3: Normalization + smoothing
        normalizer = ScoreNormalizer(method="robust", gamma=3.0)
        smoother = BandwidthMatchedSmoother(fps=fps)

        clip_norm = normalizer.normalize(clip_raw)
        ovd_norm = normalizer.normalize(ovd_raw)

        clip_smooth = smoother.smooth_signal(clip_norm, EXPERT_MODALITY_MAP["CLIP"])
        ovd_smooth = smoother.smooth_signal(ovd_norm, EXPERT_MODALITY_MAP["OVD"])

        frame_scores = {0: clip_smooth, 1: ovd_smooth}

        # Stage 4: Logic composition
        engine = LogicEngine(kappa=2.0, fps=fps, and_mode="product")
        truth_curve, _ = engine.evaluate_tree_rescaled(
            tree, frame_scores, lo=0.5, hi=1.0
        )

        assert truth_curve.shape == (num_frames,)
        assert truth_curve.min() >= 0.0
        assert truth_curve.max() <= 1.0
        # The overlap region should have the highest scores
        overlap_mean = truth_curve[42:48].mean()
        non_overlap_mean = np.concatenate([truth_curve[:30], truth_curve[60:]]).mean()
        assert overlap_mean > non_overlap_mean

        # Stage 5: Frame selection
        selector = create_selector(mode="pass")
        top_indices = selector.select(truth_curve, top_k=16, fps=fps)

        assert len(top_indices) == 16
        # Best frame should be in the overlap region
        assert 40 <= top_indices[0] <= 50

    def test_or_tree(self):
        num_frames = 50
        tree = {
            "op": "OR",
            "children": [
                {"op": "LEAF", "expert": "CLIP", "query": "a"},
                {"op": "LEAF", "expert": "CLIP", "query": "b"},
            ],
        }

        a = np.zeros(num_frames, dtype=np.float32)
        a[10] = 1.0
        b = np.zeros(num_frames, dtype=np.float32)
        b[30] = 1.0

        normalizer = ScoreNormalizer(method="robust", gamma=3.0)
        joint = normalizer.normalize_joint({"a": a, "b": b})

        frame_scores = {0: joint["a"], 1: joint["b"]}

        engine = LogicEngine(kappa=2.0, fps=1.0, and_mode="product")
        truth_curve, _ = engine.evaluate_tree(tree, frame_scores)

        # OR should have peaks at both locations
        assert truth_curve[10] > truth_curve[20]
        assert truth_curve[30] > truth_curve[20]

    def test_seq_tree(self):
        num_frames = 50
        tree = {
            "op": "SEQ",
            "children": [
                {"op": "LEAF", "expert": "CLIP", "query": "first"},
                {"op": "LEAF", "expert": "CLIP", "query": "second"},
            ],
        }

        first = np.zeros(num_frames, dtype=np.float32)
        first[10] = 1.0
        second = np.zeros(num_frames, dtype=np.float32)
        second[30] = 1.0

        normalizer = ScoreNormalizer(method="robust", gamma=3.0)
        first_n = normalizer.normalize(first)
        second_n = normalizer.normalize(second)

        frame_scores = {0: first_n, 1: second_n}

        engine = LogicEngine(kappa=2.0, fps=1.0, and_mode="product")
        truth_curve, _ = engine.evaluate_tree(tree, frame_scores)

        assert truth_curve.shape == (num_frames,)
