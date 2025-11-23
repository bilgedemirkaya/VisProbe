"""
This script is the main entry point for the VisProbe Streamlit dashboard.

Its primary role is to orchestrate the UI by calling helper functions from
`dashboard_helpers.py` to render different components. This keeps the main
application logic clean and readable.
"""

import os
import sys

import streamlit as st

from visprobe.cli.dashboard_helpers import (
    get_results,
    render_all_reports,
    render_sidebar,
)

# --- Page Configuration ---
st.set_page_config(page_title="VisProbe Dashboard", page_icon="🔬", layout="wide")


def main(module_path: str):
    """
    The main execution flow of the dashboard.

    Args:
        module_path: The path to the test module provided via command line.
    """
    st.title("🔬 VisProbe Adversarial Test Dashboard")
    st.markdown(f"**Test Module:** `{os.path.basename(module_path)}`")

    # Ensure crisp scaling for low-resolution datasets (e.g., CIFAR-10)
    st.markdown(
        """
        <style>
        img.pixelated {
          image-rendering: pixelated;
          image-rendering: -moz-crisp-edges;
          image-rendering: crisp-edges;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    results = get_results(module_path)

    if not results:
        st.error(f"No result files found for module: `{os.path.basename(module_path)}`")
        st.info(f"Please run your tests first, e.g., `python {os.path.basename(module_path)}`")
        st.stop()

    # Device banner (pull from any report that has run_meta.device)
    try:
        any_report = next(iter(results.values()))
        device = (any_report.get("run_meta") or {}).get("device")
        if device:
            st.info(f"Running on device: `{device}`")
    except Exception:
        pass

    render_sidebar(results)
    render_all_reports(results)


if __name__ == "__main__":
    # When run by streamlit, the script path is the first arg, and the user's
    # test file path is the second one.
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        st.error("No test module specified.")
        st.info("Usage: streamlit run dashboard.py -- <path_to_your_test_file.py>")
