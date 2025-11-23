"""
Pytest configuration and shared fixtures for VisProbe tests.
"""

import os
import tempfile
from typing import Tuple

import pytest
import torch
import torch.nn as nn


@pytest.fixture
def device():
    """Returns the best available device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def simple_model():
    """Returns a simple neural network for testing."""

    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(3 * 32 * 32, 128)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    model = SimpleNet()
    model.eval()
    return model


@pytest.fixture
def sample_batch(device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns a sample batch of images and labels."""
    batch_size = 4
    images = torch.randn(batch_size, 3, 32, 32, device=device)
    labels = torch.randint(0, 10, (batch_size,), device=device)
    return images, labels


@pytest.fixture
def sample_logits(device) -> torch.Tensor:
    """Returns sample logits for testing properties."""
    return torch.randn(4, 10, device=device)


@pytest.fixture
def temp_results_dir():
    """Creates a temporary directory for test results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        old_env = os.environ.get("VISPROBE_RESULTS_DIR")
        os.environ["VISPROBE_RESULTS_DIR"] = tmpdir
        yield tmpdir
        if old_env is not None:
            os.environ["VISPROBE_RESULTS_DIR"] = old_env
        elif "VISPROBE_RESULTS_DIR" in os.environ:
            del os.environ["VISPROBE_RESULTS_DIR"]


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the test registry before each test."""
    from visprobe.api.registry import TestRegistry

    TestRegistry.clear_all()
    yield
    TestRegistry.clear_all()


@pytest.fixture
def mock_art_available(monkeypatch):
    """Mock ART availability for testing."""
    import visprobe.strategies.adversarial as adv_module

    monkeypatch.setattr(adv_module, "_ART_AVAILABLE", True)
    return True
