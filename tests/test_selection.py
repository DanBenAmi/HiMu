"""Tests for frame selection strategies."""

import numpy as np
import pytest

from himu.selection import (
    ScoreRankedSelection,
    PeakBasedSelection,
    TieredPeakSelection,
    create_selector,
)


class TestScoreRankedSelection:
    def test_basic_selection(self):
        selector = ScoreRankedSelection()
        scores = np.array([0.1, 0.9, 0.3, 0.7, 0.5])
        indices = selector.select(scores, top_k=3, fps=1.0)
        assert len(indices) == 3
        assert indices[0] == 1  # highest score first

    def test_selects_all_if_k_exceeds_frames(self):
        selector = ScoreRankedSelection()
        scores = np.array([0.5, 0.3, 0.8])
        indices = selector.select(scores, top_k=10, fps=1.0)
        assert len(indices) == 3

    def test_with_min_gap(self):
        selector = ScoreRankedSelection(min_gap=2)
        scores = np.array([0.1, 0.9, 0.85, 0.8, 0.1, 0.7, 0.1])
        indices = selector.select(scores, top_k=3, fps=1.0)
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                assert abs(indices[i] - indices[j]) >= 2


class TestPeakBasedSelection:
    def test_finds_peaks(self):
        selector = PeakBasedSelection(prominence=0.01)
        scores = np.array([0.0, 0.5, 0.0, 0.0, 0.8, 0.0, 0.0, 0.3, 0.0])
        indices = selector.select(scores, top_k=3, fps=1.0)
        assert 4 in indices  # highest peak
        assert 1 in indices  # second peak

    def test_fills_to_k(self):
        selector = PeakBasedSelection()
        scores = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
        indices = selector.select(scores, top_k=3, fps=1.0)
        assert len(indices) == 3


class TestTieredPeakSelection:
    def test_basic(self):
        selector = TieredPeakSelection()
        scores = np.zeros(100)
        scores[20] = 1.0
        scores[50] = 0.8
        scores[80] = 0.6
        indices = selector.select(scores, top_k=5, fps=1.0)
        assert len(indices) == 5


class TestFactory:
    def test_score_ranked(self):
        s = create_selector(mode="score_ranked")
        assert isinstance(s, ScoreRankedSelection)

    def test_peak_nms(self):
        s = create_selector(mode="peak_nms")
        assert isinstance(s, PeakBasedSelection)

    def test_peak_nms_v2(self):
        s = create_selector(mode="peak_nms_v2")
        assert isinstance(s, TieredPeakSelection)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            create_selector(mode="unknown")
