#!/usr/bin/env python3
"""
Quick test script for the new quick_check() API.

This tests that the basic functionality works without errors.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from visprobe import quick_check

print("=" * 70)
print("Testing VisProbe quick_check() API")
print("=" * 70)

# 1. Create a simple model (ResNet-18, pretrained)
print("\n1. Loading model...")
model = models.resnet18(pretrained=False)  # Use pretrained=False for faster loading
model.eval()
print(f"   ✓ Model loaded: {model.__class__.__name__}")

# 2. Create a small dummy dataset (10 random images, 32x32)
print("\n2. Creating test data...")
num_samples = 10
test_images = torch.randn(num_samples, 3, 224, 224)  # ResNet expects 224x224
test_labels = torch.randint(0, 1000, (num_samples,))  # ImageNet classes

# Package as list of tuples
test_data = [(img, int(label.item())) for img, label in zip(test_images, test_labels)]
print(f"   ✓ Created {num_samples} test samples")

# 3. Run quick_check with "lighting" preset (simplest/fastest)
print("\n3. Running quick_check()...")
print("   (This may take 1-2 minutes...)")

try:
    report = quick_check(
        model=model,
        data=test_data,
        preset="lighting",  # Fastest preset
        budget=50,  # Small budget for quick test
        device="cpu",  # Use CPU for compatibility
    )

    print("\n" + "=" * 70)
    print("✅ SUCCESS! quick_check() ran without errors")
    print("=" * 70)

    # 4. Test the new Report methods
    print("\n4. Testing Report methods...")

    print(f"\n   report.score: {report.score}")
    print(f"   report.failures: {len(report.failures)} failures")
    print(f"   report.summary: {report.summary}")

    # Test show() method
    print("\n5. Testing report.show()...")
    report.show()

    # Test export_failures()
    print("\n6. Testing report.export_failures()...")
    if report.failures:
        export_path = report.export_failures(n=5)
        print(f"   ✓ Exported to: {export_path}")
    else:
        print("   (No failures to export)")

    print("\n" + "=" * 70)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Run preset validation (manually review images)")
    print("  2. Test on real models (CIFAR-10, ImageNet)")
    print("  3. Write compelling README")
    print("  4. Create Jupyter notebook example")

except Exception as e:
    print("\n" + "=" * 70)
    print("❌ ERROR: quick_check() failed")
    print("=" * 70)
    print(f"\n{type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
