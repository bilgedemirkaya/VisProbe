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
    render_sidebar,
    render_executive_summary,
    render_failure_triage,
    render_root_cause_analysis,
    render_search_path_insights,
    render_actionable_recommendations,
    render_key_metrics,
    render_strategies_section,
    render_image,
    render_analysis_tabs,
    _get_image_bytes,
)

# --- Page Configuration ---
st.set_page_config(page_title="VisProbe Dashboard", page_icon="🔬", layout="wide")


def render_all_reports_new(results: dict):
    """Render all reports using the new 5-section structure."""
    import io
    from PIL import Image as PILImage

    for test_name, report in results.items():
        with st.expander(f"📋 Test: `{test_name}`", expanded=True):
            # ====== SECTION 1: EXECUTIVE SUMMARY ======
            render_executive_summary(report)
            st.markdown("---")

            # ====== SECTION 2: FAILURE TRIAGE ======
            render_failure_triage(report)
            st.markdown("---")

            # ====== SECTION 3: ROOT CAUSE ANALYSIS ======
            render_root_cause_analysis(report)
            st.markdown("---")

            # ====== SECTION 4: SEARCH PATH ANALYSIS ======
            if report.get("search_path"):
                render_search_path_insights(report)
                st.markdown("---")

            # ====== SECTION 5: ACTIONABLE RECOMMENDATIONS ======
            render_actionable_recommendations(report)
            st.markdown("---")

            # ====== ADDITIONAL SECTIONS ======
            st.header("📋 Test Details")

            tab1, tab2, tab3 = st.tabs(
                ["Metrics & Strategy", "Visual Comparison", "Detailed Analysis"]
            )

            with tab1:
                st.subheader("Key Metrics")
                render_key_metrics(report)
                st.markdown("---")
                st.subheader("Applied Strategies")
                render_strategies_section(report)

            with tab2:
                st.subheader("Visual Comparison")
                try:
                    o = report.get("original_image") or {}
                    p = report.get("perturbed_image") or {}
                    ob = _get_image_bytes(o) if o else None
                    pb = _get_image_bytes(p) if p else None
                    ow = oh = pw = ph = None
                    if ob:
                        oi = PILImage.open(io.BytesIO(ob))
                        ow, oh = oi.size
                    if pb:
                        pi = PILImage.open(io.BytesIO(pb))
                        pw, ph = pi.size
                    parts = []
                    if ow and oh:
                        parts.append(f"Original: {ow}×{oh}px")
                    if pw and ph:
                        parts.append(f"Perturbed: {pw}×{ph}px")
                    if parts:
                        st.info("; ".join(parts) + ". Small images may look blocky when enlarged.")
                except Exception:
                    pass

                has_residual = report.get("residual_image") is not None
                cols = st.columns(3 if has_residual else 2)
                with cols[0]:
                    st.subheader("Original Image")
                    render_image(report.get("original_image"), "Original Image")
                with cols[1]:
                    st.subheader("Perturbed Image")
                    render_image(report.get("perturbed_image"), "Perturbed Image")
                if has_residual:
                    with cols[2]:
                        st.subheader("Residual")
                        render_image(report.get("residual_image"), "Residual")

            with tab3:
                st.subheader("Detailed Analysis")
                render_analysis_tabs(report)


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
    render_all_reports_new(results)


if __name__ == "__main__":
    # When run by streamlit, the script path is the first arg, and the user's
    # test file path is the second one.
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        st.error("No test module specified.")
        st.info("Usage: streamlit run dashboard.py -- <path_to_your_test_file.py>")
