"""
This module contains the main entry point and logic for the VisProbe CLI.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile


def get_results_dir() -> str:
    """Returns the results directory path (cross-platform)."""
    env_dir = os.environ.get("VISPROBE_RESULTS_DIR")
    if env_dir:
        return os.path.abspath(env_dir)
    return os.path.join(tempfile.gettempdir(), "visprobe_results")


def _check_for_results(module_path: str) -> bool:
    """Checks if any .json result files exist for a given test module."""
    results_dir = get_results_dir()
    module_name = os.path.splitext(os.path.basename(module_path))[0]
    if not os.path.exists(results_dir):
        return False

    for f in os.listdir(results_dir):
        if f.startswith(f"{module_name}.") and f.endswith(".json"):
            return True
    return False


def _run_test_file(test_file_path: str, device: str | None = None):
    """Executes a given Python test file as a subprocess."""
    print(f"🔬 No results found. Running test file first: {os.path.basename(test_file_path)}")

    # Get the correct module name from the file path
    module_name = os.path.splitext(os.path.basename(test_file_path))[0]

    # Set an environment variable so the runner can save the report with the correct name
    env = os.environ.copy()
    env["VISPROBE_MODULE_NAME"] = module_name
    if device:
        env["VISPROBE_DEVICE"] = device

    try:
        subprocess.run([sys.executable, test_file_path], check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"❌ Test file exited with an error: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"❌ Error: '{sys.executable}' not found. Could not run the test file.")
        sys.exit(1)


def run_command(args: argparse.Namespace):
    """Handles the 'visprobe run' command."""
    test_file = os.path.abspath(args.test_file)
    if not os.path.exists(test_file):
        print(f"❌ Error: Test file not found at '{test_file}'")
        sys.exit(1)

    results_dir = get_results_dir()
    if not args.keep and os.path.isdir(results_dir):
        for f in os.listdir(results_dir):
            if f.endswith(".json"):
                os.remove(os.path.join(results_dir, f))
        print("🗑️  Old result files removed.")

    _run_test_file(test_file, device=args.device)
    print("✅ Test run finished. Report saved.")


def visualize_command(args: argparse.Namespace):
    """Handles the 'visprobe visualize' command."""
    test_file = os.path.abspath(args.test_file)
    if not os.path.exists(test_file):
        print(f"❌ Error: Test file not found at '{test_file}'")
        sys.exit(1)

    # Automatically run the test if no results are found
    if not _check_for_results(test_file):
        _run_test_file(test_file)

    cli_dir = os.path.dirname(__file__)
    dashboard_path = os.path.join(cli_dir, "dashboard.py")

    if not os.path.exists(dashboard_path):
        print(f"❌ Error: Dashboard script not found at '{dashboard_path}'")
        sys.exit(1)

    print("🚀 Launching dashboard…")

    command = ["streamlit", "run", dashboard_path, "--", test_file]

    try:
        subprocess.run(command, check=True, capture_output=False)
    except FileNotFoundError:
        print("❌ Error: 'streamlit' command not found.")
        print("   Please make sure Streamlit is installed and in your PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ Dashboard exited with an error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user.")
        sys.exit(0)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="VisProbe: A toolbox for interactive robustness testing."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- visualize ---
    vis_parser = subparsers.add_parser(
        "visualize", help="Visualize test results in an interactive dashboard."
    )
    vis_parser.add_argument("test_file", help="Path to the Python file containing VisProbe tests.")
    vis_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps", "hip"],
        default="auto",
        help="Device to run models on (overrides VISPROBE_DEVICE). 'hip' for AMD ROCm.",
    )
    vis_parser.set_defaults(func=visualize_command)

    # --- run ---
    run_parser = subparsers.add_parser(
        "run", help="Run a VisProbe test file after clearing old result JSONs."
    )
    run_parser.add_argument("test_file", help="Path to the Python file containing VisProbe tests.")
    run_parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep previous JSON result files instead of deleting them.",
    )
    run_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps", "hip"],
        default="auto",
        help="Device to run models on (overrides VISPROBE_DEVICE). 'hip' for AMD ROCm.",
    )
    run_parser.set_defaults(func=run_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
