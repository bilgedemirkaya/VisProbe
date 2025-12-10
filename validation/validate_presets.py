#!/usr/bin/env python3
"""
Preset Validation Script

This script generates side-by-side comparison images for manual validation
of preset parameter ranges. The goal is to ensure that perturbed images
still preserve the original label (i.e., a human can still recognize them).

Usage:
    python validation/validate_presets.py [--preset PRESET_NAME] [--num-samples N]

After running, manually review the images in validation/output/ and adjust
preset ranges in src/visprobe/presets.py if needed.

Target: ~85-90% of perturbed images should still be recognizable.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid

from visprobe.presets import PRESETS, get_preset
from visprobe.strategies.image import (
    BrightnessStrategy,
    ContrastStrategy,
    GammaStrategy,
    GaussianBlurStrategy,
    GaussianNoiseStrategy,
    JPEGCompressionStrategy,
    MotionBlurStrategy,
)

# CIFAR-10 class names for reference
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def instantiate_strategy(strategy_config):
    """Instantiate a strategy from config dict."""
    strategy_type = strategy_config["type"]

    if strategy_type == "brightness":
        return BrightnessStrategy(brightness_factor=1.0)
    elif strategy_type == "contrast":
        return ContrastStrategy(contrast_factor=1.0)
    elif strategy_type == "gamma":
        return GammaStrategy(gamma=1.0)
    elif strategy_type == "gaussian_blur":
        kernel_size = strategy_config.get("kernel_size", 5)
        return GaussianBlurStrategy(kernel_size=kernel_size, sigma=0.0)
    elif strategy_type == "motion_blur":
        angle = strategy_config.get("angle", 0.0)
        return MotionBlurStrategy(kernel_size=1, angle=angle)
    elif strategy_type == "jpeg_compression":
        return JPEGCompressionStrategy(quality=100)
    elif strategy_type == "gaussian_noise":
        return GaussianNoiseStrategy(std_dev=0.0)
    elif strategy_type == "compositional":
        # Skip compositional for now - validate individual ones first
        return None
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")


def get_level_bounds(strategy_config):
    """Extract min/max levels from strategy config."""
    strategy_type = strategy_config["type"]

    if strategy_type == "brightness":
        return (strategy_config.get("min_factor", 0.5),
                strategy_config.get("max_factor", 1.5))
    elif strategy_type == "contrast":
        return (strategy_config.get("min_factor", 0.7),
                strategy_config.get("max_factor", 1.3))
    elif strategy_type == "gamma":
        return (strategy_config.get("min_gamma", 0.7),
                strategy_config.get("max_gamma", 1.3))
    elif strategy_type == "gaussian_blur":
        return (strategy_config.get("min_sigma", 0.0),
                strategy_config.get("max_sigma", 2.5))
    elif strategy_type == "motion_blur":
        return (strategy_config.get("min_kernel", 1),
                strategy_config.get("max_kernel", 25))
    elif strategy_type == "jpeg_compression":
        return (strategy_config.get("min_quality", 10),
                strategy_config.get("max_quality", 100))
    elif strategy_type == "gaussian_noise":
        return (strategy_config.get("min_std", 0.0),
                strategy_config.get("max_std", 0.05))
    else:
        return (0.0, 1.0)


def denormalize_for_display(tensor):
    """Convert normalized tensor to displayable format."""
    # Assuming tensor is in [0, 1] range already
    tensor = tensor.clone()
    tensor = torch.clamp(tensor, 0, 1)
    return tensor


def create_comparison_grid(original_images, perturbed_images_dict, labels, output_path):
    """
    Create a comparison grid showing original vs perturbed images.

    Args:
        original_images: Tensor of original images (N, C, H, W)
        perturbed_images_dict: Dict of {level: perturbed_tensor}
        labels: List of label names
        output_path: Where to save the image
    """
    num_images = original_images.shape[0]
    num_levels = len(perturbed_images_dict)

    # Create figure
    fig, axes = plt.subplots(
        num_images,
        num_levels + 1,  # +1 for original
        figsize=(3 * (num_levels + 1), 3 * num_images)
    )

    if num_images == 1:
        axes = axes.reshape(1, -1)

    levels = sorted(perturbed_images_dict.keys())

    for i in range(num_images):
        # Original image
        ax = axes[i, 0]
        img = denormalize_for_display(original_images[i]).permute(1, 2, 0).cpu().numpy()
        ax.imshow(img, interpolation='nearest')  # No blurring interpolation
        ax.set_title(f"Original\n{labels[i]}", fontsize=10)
        ax.axis('off')

        # Perturbed images at different levels
        for j, level in enumerate(levels, 1):
            ax = axes[i, j]
            perturbed = perturbed_images_dict[level][i]
            img = denormalize_for_display(perturbed).permute(1, 2, 0).cpu().numpy()
            ax.imshow(img, interpolation='nearest')  # No blurring interpolation
            ax.set_title(f"Level: {level:.2f}", fontsize=10)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')  # Higher DPI for sharper images
    plt.close()

    print(f"  ✓ Saved: {output_path}")


def validate_preset(preset_name, num_samples=10, output_dir="validation/output"):
    """
    Validate a preset by generating comparison images.

    Args:
        preset_name: Name of preset to validate
        num_samples: Number of test samples to use
        output_dir: Output directory for validation images
    """
    print(f"\n{'='*70}")
    print(f"Validating Preset: {preset_name.upper()}")
    print(f"{'='*70}")

    # Load preset
    preset = get_preset(preset_name)
    print(f"\nPreset: {preset['name']}")
    print(f"Description: {preset['description']}")

    # Create output directory
    preset_output_dir = Path(output_dir) / preset_name
    preset_output_dir.mkdir(parents=True, exist_ok=True)

    # Load test data (CIFAR-10)
    print(f"\nLoading {num_samples} CIFAR-10 test samples...")
    transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),  # Higher resolution for validation
        T.ToTensor(),
    ])

    dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Select diverse samples (one from each class if possible)
    samples_per_class = max(1, num_samples // 10)
    test_images = []
    test_labels = []

    for class_idx in range(10):
        class_samples = [i for i, (_, label) in enumerate(dataset) if label == class_idx]
        selected = class_samples[:samples_per_class]
        for idx in selected:
            img, label = dataset[idx]
            test_images.append(img)
            test_labels.append(CIFAR10_CLASSES[label])
            if len(test_images) >= num_samples:
                break
        if len(test_images) >= num_samples:
            break

    test_images = torch.stack(test_images)
    print(f"  ✓ Loaded {len(test_images)} samples")

    # Validate each strategy in the preset
    print(f"\nValidating {len(preset['strategies'])} strategies...")

    validation_results = []

    for idx, strategy_config in enumerate(preset['strategies'], 1):
        strategy_type = strategy_config['type']

        # Skip compositional for individual validation
        if strategy_type == "compositional":
            print(f"\n  {idx}. Skipping compositional strategy (validate components first)")
            continue

        print(f"\n  {idx}. Validating: {strategy_type}")

        # Instantiate strategy
        strategy = instantiate_strategy(strategy_config)
        if strategy is None:
            continue

        # Get level bounds
        level_min, level_max = get_level_bounds(strategy_config)
        print(f"     Range: [{level_min:.3f}, {level_max:.3f}]")

        # Generate images at different severity levels
        test_levels = [
            level_min,                                      # Minimum
            level_min + (level_max - level_min) * 0.33,   # 33%
            level_min + (level_max - level_min) * 0.67,   # 67%
            level_max,                                      # Maximum
        ]

        perturbed_images_dict = {}

        for level in test_levels:
            # Apply perturbation
            perturbed = strategy.generate(
                test_images.clone(),
                model=None,  # Model not needed for natural perturbations
                level=level
            )
            perturbed_images_dict[level] = perturbed

        # Create comparison grid
        output_filename = f"{idx:02d}_{strategy_type}_comparison.png"
        output_path = preset_output_dir / output_filename

        create_comparison_grid(
            test_images,
            perturbed_images_dict,
            test_labels,
            output_path
        )

        validation_results.append({
            'strategy': strategy_type,
            'range': (level_min, level_max),
            'output_file': output_filename
        })

    # Create validation report
    report_path = preset_output_dir / "VALIDATION_REPORT.md"
    create_validation_report(preset_name, preset, validation_results, report_path)

    print(f"\n{'='*70}")
    print(f"✅ Validation complete for '{preset_name}'")
    print(f"{'='*70}")
    print(f"\n📁 Output directory: {preset_output_dir}")
    print(f"📄 Validation report: {report_path}")
    print(f"\n👁️  NEXT STEPS:")
    print(f"   1. Open the output directory and review all images")
    print(f"   2. For each perturbation, ask: 'Can I still recognize the object?'")
    print(f"   3. If >15% of images are unrecognizable, reduce the max range")
    print(f"   4. Update ranges in src/visprobe/presets.py")
    print(f"   5. Re-run validation to verify improvements")
    print(f"\n   Target: 85-90% of perturbed images should be recognizable")


def create_validation_report(preset_name, preset, validation_results, output_path):
    """Create a markdown validation report."""
    with open(output_path, 'w') as f:
        f.write(f"# Validation Report: {preset['name']}\n\n")
        f.write(f"**Preset:** `{preset_name}`\n\n")
        f.write(f"**Description:** {preset['description']}\n\n")
        f.write(f"**Generated:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("---\n\n")
        f.write("## Validation Instructions\n\n")
        f.write("For each image comparison:\n\n")
        f.write("1. Look at the **original image** (leftmost column)\n")
        f.write("2. Identify what object/class it is\n")
        f.write("3. Look at each **perturbed version** (other columns)\n")
        f.write("4. Ask: **Can I still recognize this as the same object?**\n\n")

        f.write("### Acceptance Criteria\n\n")
        f.write("- ✅ **PASS**: 85-90%+ of perturbed images are still recognizable\n")
        f.write("- ⚠️  **MARGINAL**: 70-85% recognizable (consider reducing max range)\n")
        f.write("- ❌ **FAIL**: <70% recognizable (MUST reduce max range)\n\n")

        f.write("---\n\n")
        f.write("## Strategies Validated\n\n")

        for i, result in enumerate(validation_results, 1):
            f.write(f"### {i}. {result['strategy'].replace('_', ' ').title()}\n\n")
            f.write(f"**Range:** `[{result['range'][0]:.3f}, {result['range'][1]:.3f}]`\n\n")
            f.write(f"**Image:** `{result['output_file']}`\n\n")
            f.write(f"**Manual Review:**\n\n")
            f.write(f"- [ ] Level 1 ({result['range'][0]:.3f}): % recognizable = ____\n")
            f.write(f"- [ ] Level 2: % recognizable = ____\n")
            f.write(f"- [ ] Level 3: % recognizable = ____\n")
            f.write(f"- [ ] Level 4 ({result['range'][1]:.3f}): % recognizable = ____\n\n")
            f.write(f"**Overall Assessment:**\n\n")
            f.write(f"- [ ] ✅ PASS (keep current range)\n")
            f.write(f"- [ ] ⚠️  MARGINAL (consider adjustment)\n")
            f.write(f"- [ ] ❌ FAIL (reduce max range to: _____)\n\n")
            f.write(f"**Notes:**\n\n")
            f.write(f"_[Add any observations here]_\n\n")
            f.write("---\n\n")

        f.write("## Summary\n\n")
        f.write("**Overall Preset Assessment:**\n\n")
        f.write("- [ ] ✅ All strategies validated - preset ready for use\n")
        f.write("- [ ] ⚠️  Some adjustments needed\n")
        f.write("- [ ] ❌ Major adjustments required\n\n")
        f.write("**Recommended Actions:**\n\n")
        f.write("_[List any preset range adjustments to make in presets.py]_\n\n")
        f.write("---\n\n")
        f.write("**Validator:** _[Your name]_\n\n")
        f.write("**Date:** _[Date completed]_\n")

    print(f"  ✓ Created validation report: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate VisProbe preset configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a specific preset
  python validation/validate_presets.py --preset lighting

  # Validate all presets
  python validation/validate_presets.py --all

  # Validate with more samples
  python validation/validate_presets.py --preset standard --num-samples 20
        """
    )

    parser.add_argument(
        '--preset',
        type=str,
        choices=['standard', 'lighting', 'blur', 'corruption'],
        help='Preset to validate'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Validate all presets'
    )

    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Number of test samples to use (default: 10)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='validation/output',
        help='Output directory for validation images'
    )

    args = parser.parse_args()

    # Determine which presets to validate
    if args.all:
        presets_to_validate = ['standard', 'lighting', 'blur', 'corruption']
    elif args.preset:
        presets_to_validate = [args.preset]
    else:
        print("Error: Must specify --preset or --all")
        parser.print_help()
        return

    print("\n" + "="*70)
    print("VisProbe Preset Validation")
    print("="*70)
    print(f"\nPresets to validate: {', '.join(presets_to_validate)}")
    print(f"Samples per preset: {args.num_samples}")
    print(f"Output directory: {args.output_dir}")

    # Validate each preset
    for preset_name in presets_to_validate:
        validate_preset(
            preset_name,
            num_samples=args.num_samples,
            output_dir=args.output_dir
        )

    # Final summary
    print("\n" + "="*70)
    print("🎉 ALL VALIDATIONS COMPLETE")
    print("="*70)
    print(f"\n📁 All validation images saved to: {args.output_dir}/")
    print(f"\n👁️  MANUAL REVIEW REQUIRED:")
    print(f"   1. Open {args.output_dir}/")
    print(f"   2. Review each preset's comparison images")
    print(f"   3. Fill out the VALIDATION_REPORT.md in each preset folder")
    print(f"   4. Adjust ranges in src/visprobe/presets.py if needed")
    print(f"   5. Update VALIDATION_STATUS in presets.py")
    print(f"\n✨ Once validated, update presets.py:")
    print(f"   VALIDATION_STATUS['{presets_to_validate[0]}']['validated'] = True")
    print(f"   VALIDATION_STATUS['{presets_to_validate[0]}']['validation_date'] = '2024-12-09'")
    print(f"   VALIDATION_STATUS['{presets_to_validate[0]}']['label_preservation_rate'] = 0.87  # Your measured rate")


if __name__ == "__main__":
    main()
