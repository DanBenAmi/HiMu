"""Tests for bandwidth-matched smoothing."""

import numpy as np
import pytest

from himu.smoothing import BandwidthMatchedSmoother, EXPERT_MODALITY_MAP


class TestBandwidthMatchedSmoother:
    def test_visual_smoothing(self):
        smoother = BandwidthMatchedSmoother(fps=1.0)
        signal = np.zeros(50)
        signal[25] = 1.0  # impulse
        result = smoother.smooth_signal(signal, "visual")
        assert result[25] < 1.0  # peak reduced by smoothing
        assert result[24] > 0.0  # neighbors lifted

    def test_speech_wider_than_visual(self):
        smoother = BandwidthMatchedSmoother(fps=1.0)
        signal = np.zeros(100)
        signal[50] = 1.0
        visual = smoother.smooth_signal(signal.copy(), "visual")
        speech = smoother.smooth_signal(signal.copy(), "speech")
        # Speech smoothing is wider, so peak should be lower
        assert speech[50] < visual[50]

    def test_audio_wider_than_speech(self):
        smoother = BandwidthMatchedSmoother(fps=1.0)
        signal = np.zeros(100)
        signal[50] = 1.0
        speech = smoother.smooth_signal(signal.copy(), "speech")
        audio = smoother.smooth_signal(signal.copy(), "audio")
        assert audio[50] < speech[50]

    def test_custom_sigmas(self):
        smoother = BandwidthMatchedSmoother(
            sigmas={"visual": 1.0, "speech": 2.0, "audio": 3.0},
            fps=2.0,
        )
        signal = np.zeros(50)
        signal[25] = 1.0
        result = smoother.smooth_signal(signal, "visual")
        assert result[25] < 1.0

    def test_preserves_shape(self):
        smoother = BandwidthMatchedSmoother(fps=1.0)
        signal = np.random.rand(30)
        result = smoother.smooth_signal(signal, "visual")
        assert result.shape == signal.shape

    def test_unknown_modality_raises(self):
        smoother = BandwidthMatchedSmoother(fps=1.0)
        with pytest.raises(ValueError):
            smoother.smooth_signal(np.zeros(10), "unknown")


class TestExpertModalityMap:
    def test_visual_experts(self):
        assert EXPERT_MODALITY_MAP["CLIP"] == "visual"
        assert EXPERT_MODALITY_MAP["OVD"] == "visual"
        assert EXPERT_MODALITY_MAP["OCR"] == "visual"

    def test_audio_experts(self):
        assert EXPERT_MODALITY_MAP["ASR"] == "speech"
        assert EXPERT_MODALITY_MAP["CLAP"] == "audio"
