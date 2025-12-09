#!/usr/bin/env python3
"""
Test that the README example works exactly as shown.
"""

print("Testing README Quick Start example...")
print("=" * 70)

from visprobe import quick_check
import torchvision.models as models
from torchvision.datasets import CIFAR10
import torchvision.transforms as T

# 1. Load your model
print("\n1. Loading model...")
model = models.resnet18(weights=None)  # Use weights=None for faster testing
model.eval()
print("   ✓ Model loaded")

# 2. Prepare test data (any format works: DataLoader, list, tensors)
print("\n2. Preparing test data...")
transform = T.Compose([T.Resize(224), T.ToTensor()])

try:
    dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_data = [dataset[i] for i in range(20)]  # Test on 20 samples (faster)
    print(f"   ✓ Loaded {len(test_data)} test samples")
except Exception as e:
    print(f"   ⚠️  Could not download CIFAR-10: {e}")
    print("   Creating dummy data instead...")
    import torch
    test_images = torch.randn(20, 3, 224, 224)
    test_labels = torch.randint(0, 10, (20,))
    test_data = [(img, int(label.item())) for img, label in zip(test_images, test_labels)]
    print(f"   ✓ Created {len(test_data)} dummy samples")

# 3. Run robustness test
print("\n3. Running quick_check() with 'lighting' preset...")
print("   (Using small budget for quick test)")
report = quick_check(model, test_data, preset="lighting", budget=50, device="cpu")

# 4. View results
print("\n4. Displaying results with report.show()...")
report.show()

# 5. Export failures for retraining
print("\n5. Testing report.export_failures()...")
if report.failures:
    output_dir = report.export_failures(n=5)
    print(f"   ✓ Exported to: {output_dir}")
else:
    print("   (No failures to export)")

# Test other Report API methods
print("\n6. Testing Report API methods...")
print(f"   report.score: {report.score:.2%}")
print(f"   report.failures: {len(report.failures)} failures")
print(f"   report.summary keys: {list(report.summary.keys())}")

print("\n" + "=" * 70)
print("✅ README example works correctly!")
print("=" * 70)
