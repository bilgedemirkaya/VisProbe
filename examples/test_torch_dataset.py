"""
This is a test that uses a torch.utils.data.Dataset as the data source.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from visprobe.api.decorators import data_source as source
from visprobe.api.decorators import given, model
from visprobe.properties.classification import ConfidenceDrop, LabelConstant
from visprobe.strategies import FGSMStrategy


# --- Configuration ---
class SimpleModel(nn.Module):
    """A simple neural network for testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class RandomDataset(Dataset):
    """A dataset that returns random data."""

    def __init__(self, num_samples=10, num_features=10):
        self.num_samples = num_samples
        self.num_features = num_features

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.randn(self.num_features)


model_instance = SimpleModel()
dataset_instance = RandomDataset()


# --- VisProbe Test ---
@given(strategy=FGSMStrategy(eps=0.008))
@model(model_instance)
@source(dataset_instance, collate_fn=torch.stack)
def test_fgsm_attack_on_dataset(original, perturbed):
    """Test robustness to small FGSM attack (ε=8/255)."""
    assert LabelConstant.evaluate(original, perturbed), "Label changed under FGSM"
    assert ConfidenceDrop.evaluate(original, perturbed, max_drop=0.3), "Confidence drop >30%"


# --- Main Execution ---
if __name__ == "__main__":
    result = test_fgsm_attack_on_dataset()
    print("Test finished.")
    if result.failures:
        print(f"{len(result.failures)} failures found.")
    else:
        print("No failures found.")
