"""
Tests for the visprobe.properties module.
"""

import pytest
import torch

from visprobe.properties import (
    ConfidenceDrop,
    L2Distance,
    LabelConstant,
    TopKStability,
    get_top_prediction,
    get_topk_predictions,
)
from visprobe.properties.helpers import extract_logits


class TestLabelConstant:
    """Tests for LabelConstant property."""

    def test_same_label_passes(self, sample_logits):
        """Test that identical logits pass."""
        prop = LabelConstant()
        assert prop(sample_logits, sample_logits) is True

    def test_different_label_fails(self, sample_logits):
        """Test that different labels fail."""
        prop = LabelConstant()
        perturbed = sample_logits.clone()
        # Flip the top prediction
        perturbed[0, 0] = 100.0
        perturbed[0, 1:] = -100.0
        assert prop(sample_logits, perturbed) is False

    def test_tuple_input(self, sample_logits):
        """Test that tuple (logits, features) input works."""
        prop = LabelConstant()
        original = (sample_logits, torch.randn(4, 128))
        perturbed = (sample_logits, torch.randn(4, 128))
        assert prop(original, perturbed) is True


class TestTopKStability:
    """Tests for TopKStability property."""

    def test_overlap_mode(self, sample_logits):
        """Test overlap mode."""
        prop = TopKStability(k=5, mode="overlap", min_overlap=3)
        assert prop(sample_logits, sample_logits) is True

    def test_containment_mode(self, sample_logits):
        """Test containment mode."""
        prop = TopKStability(k=5, mode="containment")
        assert prop(sample_logits, sample_logits) is True

    def test_jaccard_mode(self, sample_logits):
        """Test jaccard mode."""
        prop = TopKStability(k=5, mode="jaccard", min_jaccard=0.4)
        assert prop(sample_logits, sample_logits) is True

    def test_invalid_k_raises(self):
        """Test that k < 1 raises ValueError."""
        with pytest.raises(ValueError, match="k must be >= 1"):
            TopKStability(k=0)

    def test_invalid_overlap_raises(self):
        """Test that invalid min_overlap raises ValueError."""
        with pytest.raises(ValueError, match="min_overlap must be between 1 and k"):
            TopKStability(k=5, mode="overlap", min_overlap=10)

    def test_invalid_jaccard_raises(self):
        """Test that invalid min_jaccard raises ValueError."""
        with pytest.raises(ValueError, match="min_jaccard must be in"):
            TopKStability(k=5, mode="jaccard", min_jaccard=1.5)


class TestConfidenceDrop:
    """Tests for ConfidenceDrop property."""

    def test_no_drop_passes(self, sample_logits):
        """Test that identical logits pass."""
        prop = ConfidenceDrop(max_drop=0.3)
        assert prop(sample_logits, sample_logits) is True

    def test_small_drop_passes(self, sample_logits):
        """Test that small confidence drops pass."""
        prop = ConfidenceDrop(max_drop=0.3)
        perturbed = sample_logits * 0.95  # Slight reduction
        assert prop(sample_logits, perturbed) is True

    def test_large_drop_fails(self, sample_logits):
        """Test that large confidence drops fail."""
        prop = ConfidenceDrop(max_drop=0.1)
        perturbed = sample_logits * 0.1  # Major reduction
        assert prop(sample_logits, perturbed) is False

    def test_invalid_max_drop_raises(self):
        """Test that invalid max_drop raises ValueError."""
        with pytest.raises(ValueError, match="max_drop must be between 0.0 and 1.0"):
            ConfidenceDrop(max_drop=1.5)


class TestL2Distance:
    """Tests for L2Distance property."""

    def test_zero_distance_passes(self, sample_logits):
        """Test that identical logits pass."""
        prop = L2Distance(max_delta=1.0)
        assert prop(sample_logits, sample_logits) is True

    def test_small_distance_passes(self, sample_logits):
        """Test that small perturbations pass."""
        prop = L2Distance(max_delta=10.0)
        perturbed = sample_logits + torch.randn_like(sample_logits) * 0.01
        assert prop(sample_logits, perturbed) is True

    def test_large_distance_fails(self, sample_logits):
        """Test that large perturbations fail."""
        prop = L2Distance(max_delta=0.1)
        perturbed = sample_logits + torch.randn_like(sample_logits) * 100.0
        assert prop(sample_logits, perturbed) is False


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_top_prediction(self, sample_logits):
        """Test get_top_prediction function."""
        idx, conf = get_top_prediction(sample_logits)
        assert isinstance(idx, int)
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_get_topk_predictions(self, sample_logits):
        """Test get_topk_predictions function."""
        indices, confs = get_topk_predictions(sample_logits, k=5)
        assert isinstance(indices, list)
        assert isinstance(confs, list)
        assert len(indices) == 5
        assert len(confs) == 5

    def test_get_topk_invalid_k_raises(self, sample_logits):
        """Test that k < 1 raises ValueError."""
        with pytest.raises(ValueError, match="k must be >= 1"):
            get_topk_predictions(sample_logits, k=0)

    def test_get_topk_empty_tensor_raises(self):
        """Test that empty tensor raises ValueError."""
        empty = torch.tensor([])
        with pytest.raises(ValueError, match="Cannot get top-k predictions from empty tensor"):
            get_topk_predictions(empty, k=5)

    def test_extract_logits_from_tensor(self, sample_logits):
        """Test extracting logits from raw tensor."""
        result = extract_logits(sample_logits)
        assert torch.equal(result, sample_logits)

    def test_extract_logits_from_tuple(self, sample_logits):
        """Test extracting logits from tuple."""
        features = torch.randn(4, 128)
        result = extract_logits((sample_logits, features))
        assert torch.equal(result, sample_logits)

    def test_extract_logits_from_dict(self, sample_logits):
        """Test extracting logits from dict."""
        result = extract_logits({"output": sample_logits})
        assert torch.equal(result, sample_logits)

    def test_extract_logits_invalid_dict_raises(self):
        """Test that dict without 'output' key raises TypeError."""
        with pytest.raises(TypeError, match="Dict must contain 'output' key"):
            extract_logits({"wrong_key": torch.randn(4, 10)})

    def test_extract_logits_invalid_type_raises(self):
        """Test that invalid type raises TypeError."""
        with pytest.raises(TypeError, match="Expected torch.Tensor"):
            extract_logits([1, 2, 3])
