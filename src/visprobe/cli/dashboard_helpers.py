"""
This module contains helper functions for rendering UI components in the
VisProbe Streamlit dashboard. This keeps the main dashboard script clean
and focused on the overall application flow.
"""

import base64
import io
import json
import logging
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from .utils import get_results_dir

logger = logging.getLogger(__name__)


def _get_image_bytes(image_dict: dict) -> Optional[bytes]:
    """Return raw PNG/JPEG bytes from either base64 or file path in image_dict."""
    if not image_dict:
        return None
    # Prefer path if available (JSONs saved by VisProbe omit base64 to keep files small)
    img_path = image_dict.get("image_path")
    if img_path and os.path.exists(img_path):
        try:
            with open(img_path, "rb") as f:
                return f.read()
        except OSError as e:
            logger.warning(f"Failed to read image from {img_path}: {e}")
    # Fallback to base64 if present (older reports or in-memory rendering)
    b64 = image_dict.get("image_b64")
    if b64:
        try:
            return base64.b64decode(b64)
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to decode base64 image: {e}")
            return None
    return None


def get_results(module_path: str) -> Dict[str, Any]:
    """Loads all JSON reports for a given test module."""
    results_dir = get_results_dir()
    module_name = os.path.splitext(os.path.basename(module_path))[0]
    results = {}
    if not os.path.exists(results_dir):
        return results

    for f in sorted(os.listdir(results_dir)):
        if (f.startswith(f"{module_name}.") or f.startswith("__main__.")) and f.endswith(".json"):
            try:
                test_name = ".".join(f.split(".")[1:-1])
                with open(os.path.join(results_dir, f), "r") as file:
                    results[test_name] = json.load(file)
            except (IOError, json.JSONDecodeError, IndexError):
                pass
    return results


def render_sidebar(results: dict):
    """Renders the sidebar with download buttons for each test report."""
    results_dir = get_results_dir()
    for test_name, report in results.items():
        st.sidebar.download_button(
            label=f"⬇️ Download JSON `{test_name}`",
            data=json.dumps(report, indent=2),
            file_name=f"{test_name}_report.json",
            mime="application/json",
        )
        # Optional per-sample CSV if available
        csv_path = os.path.join(results_dir, f"{test_name}.csv")
        if os.path.exists(csv_path):
            try:
                with open(csv_path, "rb") as cf:
                    st.sidebar.download_button(
                        label=f"📄 Export CSV `{test_name}`",
                        data=cf.read(),
                        file_name=f"{test_name}_per_sample.csv",
                        mime="text/csv",
                    )
            except (OSError, IOError) as e:
                logger.warning(f"Failed to read CSV file {csv_path}: {e}")


def render_image(image_data: dict, header: str, fixed_width: int = 200):
    """Renders a base64 image with crisp scaling using a pixelated CSS class."""
    if not image_data:
        st.warning(f"No {header.lower()} data available.")
        return

    data_bytes = _get_image_bytes(image_data)
    if not data_bytes:
        st.warning(f"No {header.lower()} data available.")
        return

    # Derive optional size info for caption
    width_px = height_px = None
    try:
        pil = Image.open(io.BytesIO(data_bytes))
        width_px, height_px = pil.size
    except (OSError, ValueError) as e:
        logger.debug(f"Failed to extract image dimensions: {e}")

    caption = (
        f"Prediction: {image_data.get('prediction','?')} ({image_data.get('confidence', 0):.2%})"
    )
    if width_px and height_px:
        caption += f"  |  {width_px}×{height_px}px"

    img_b64 = base64.b64encode(data_bytes).decode("utf-8")
    html = f'<img class="pixelated" src="data:image/png;base64,{img_b64}" width="{fixed_width}"/>'
    st.markdown(html, unsafe_allow_html=True)
    st.caption(caption)


def _extract_strength_info(params: Dict[str, Any]) -> str:
    """Extract strength info from strategy params (DRY helper)."""
    if not params:
        return "Default"
    if "std_dev" in params and params["std_dev"] is not None:
        return f"σ = {params['std_dev']:.4f}"
    if "eps" in params and params["eps"] is not None:
        return f"ε = {params['eps']:.4f}"
    if "brightness_factor" in params and params["brightness_factor"] is not None:
        return f"factor = {params['brightness_factor']:.3f}"
    if "sigma" in params and params["sigma"] is not None:
        return f"σ = {params['sigma']:.3f}"
    if "severity" in params and params["severity"] is not None:
        return f"quality = {params['severity']:.0f}"
    return "Default"


def _format_strategy_name(strategy_name: str) -> str:
    """Convert strategy class names to user-friendly display names."""
    name_mapping = {
        "GaussianNoiseStrategy": "Gaussian Noise",
        "FGSMStrategy": "FGSM Attack",
        "PGDStrategy": "PGD Attack",
        "BrightnessStrategy": "Brightness Adjustment",
        "ContrastStrategy": "Contrast Adjustment",
        "GaussianBlurStrategy": "Gaussian Blur",
        "JPEGCompressionStrategy": "JPEG Compression",
        "RotationStrategy": "Rotation",
        "TranslationStrategy": "Translation",
        "ScaleStrategy": "Scaling",
        "CompositeStrategy": "Multiple Strategies",
    }
    return name_mapping.get(strategy_name, strategy_name.replace("Strategy", ""))


def _get_strategy_summary(report: dict) -> str:
    """Get a brief summary of strategies for key metrics."""
    perturbation_info = report.get("perturbation_info")
    if not perturbation_info:
        return "Custom"

    name = perturbation_info.get("name", "Unknown")
    params = perturbation_info.get("params", {})

    # Handle composite strategies
    if name == "CompositeStrategy" and "components" in params:
        count = len(params["components"])
        return f"Composite ({count} strategies)"

    # Handle single strategy
    return _format_strategy_name(name)


def render_key_metrics(report: dict):
    """Renders the main summary metrics for a report."""
    st.header("📊 Key Metrics")
    cols = st.columns(5)  # Increased from 4 to 5 columns
    test_type = report.get("test_type", "N/A")

    if test_type == "given":
        accuracy = report.get("robust_accuracy")
        passed = report.get("passed_samples", 0)
        total = report.get("total_samples", 0)
        cols[0].metric(
            "Robust Accuracy",
            f"{accuracy:.1%}" if accuracy is not None else "N/A",
            f"{passed}/{total} Samples Passed",
            help="Fraction of samples whose top-1 label is unchanged under the fixed perturbation.",
        )
    elif test_type == "search":
        threshold = report.get("failure_threshold")
        cols[0].metric(
            "Failure Threshold (ε)",
            f"{threshold:.5f}" if threshold is not None else "Not Found",
            "Lower is less robust",
            help="The minimum perturbation level (ε) where the test property failed for at least one image in the batch.",  # noqa: E501
        )
        try:
            sp = report.get("search_path") or []
            if sp:
                last = sp[-1]
                pf = last.get("passed_frac")
                if pf is not None:
                    cols[2].metric(
                        "Pass Fraction (last)",
                        f"{pf:.0%}",
                        help="Share of images still satisfying the property at the last evaluated level.",  # noqa: E501
                    )
        except (KeyError, IndexError, TypeError) as e:
            logger.debug(f"Failed to extract pass fraction from search path: {e}")

        # Quantile-based batch threshold summary (if available)
        q = (report.get("aggregates") or {}).get("threshold_quantiles")
        if isinstance(q, dict):
            cols[3].metric(
                "Median Threshold",
                f"{q.get('median', float('nan')):.5f}",
                help="Median per-image failure threshold from the search path.",
            )
            cols[4].metric(
                "p05–p95",
                f"{q.get('p05', float('nan')):.5f} – {q.get('p95', float('nan')):.5f}",
                help="5th to 95th percentile of per-image thresholds (spread of robustness).",
            )

        # Add strategies metric
    strategies_summary = _get_strategy_summary(report)
    cols[1].metric(
        "Strategies",
        strategies_summary,
        help="Applied perturbations/strategies. See detailed breakdown below.",
    )

    cols[2].metric(
        "Test Type",
        test_type.capitalize(),
        help="Whether this was a fixed perturbation test or a search.",
    )
    cols[3].metric(
        "Model Queries",
        report.get("model_queries", "N/A"),
        help="Total number of forward passes, including any inside strategies/properties.",
    )
    cols[4].metric(
        "Runtime", f"{report.get('runtime', 0):.2f}s", help="Wall-clock time for the run."
    )


def render_strategies_section(report: dict):
    """Renders detailed strategies information in a dedicated section."""
    perturbation_info = report.get("perturbation_info")
    if not perturbation_info:
        return

    st.header("🎯 Applied Strategies")

    # Extract property information
    property_name = report.get("property_name") or report.get("property", "Custom")

    name = perturbation_info.get("name", "Unknown")
    params = perturbation_info.get("params", {})

    if name == "CompositeStrategy" and "components" in params:
        # Multiple strategies - show as table
        components = params["components"]

        # Create table data
        table_data = []
        for i, comp in enumerate(components, 1):
            comp_name = comp.get("name", "Unknown")
            comp_params = comp.get("params", {})

            strength = _extract_strength_info(comp_params)
            target_info = ""
            if comp_params and "targeted" in comp_params:
                target_info = "Targeted" if comp_params["targeted"] else "Untargeted"

            table_data.append(
                {
                    "Step": i,
                    "Strategy": _format_strategy_name(comp_name),
                    "Strength": strength,
                    "Type": target_info if target_info else "—",
                    "Property": property_name,
                }
            )

        # Display table with column configuration
        df = pd.DataFrame(table_data)

        # Configure columns with tooltips
        column_config = {
            "Step": st.column_config.NumberColumn(
                "Step", help="Execution order - strategies are applied sequentially", width="small"
            ),
            "Strategy": st.column_config.TextColumn(
                "Strategy", help="Type of perturbation applied to the input", width="medium"
            ),
            "Strength": st.column_config.TextColumn(
                "Strength",
                help="Perturbation intensity (σ=noise std, ε=attack budget, etc.)",
                width="small",
            ),
            "Type": st.column_config.TextColumn(
                "Type",
                help="Attack targeting: Targeted attacks aim for specific wrong class, Untargeted attacks aim for any misclassification",  # noqa: E501
                width="small",
            ),
            "Property": st.column_config.TextColumn(
                "Property",
                help="Robustness property being tested (e.g., LabelConstant, TopKStability)",
                width="medium",
            ),
        }

        st.dataframe(df, use_container_width=True, hide_index=True, column_config=column_config)

        st.info(
            "💡 **Order matters**: Strategies are applied sequentially from Step 1 to Step "
            + str(len(components))
            + "."
        )

    else:
        # Single strategy - show as enhanced info
        strategy_name = _format_strategy_name(name)
        strength_info = _extract_strength_info(params)
        target_info = ""
        if params and "targeted" in params:
            target_info = "Targeted" if params["targeted"] else "Untargeted"

        # Display in columns
        col1, col2, col3, col4 = st.columns([2, 1, 1, 2])

        with col1:
            st.metric("Strategy", strategy_name)

        with col2:
            st.metric("Strength", strength_info)

        with col3:
            if target_info:
                st.metric(
                    "Type",
                    target_info,
                    help="Attack targeting: Targeted attacks aim for specific wrong class, Untargeted attacks aim for any misclassification",  # noqa: E501
                )
            else:
                st.metric("Type", "—", help="Not applicable for non-adversarial strategies")

        with col4:
            st.metric("Property", property_name, help="Robustness property being tested")


def render_ensemble_analysis(report: dict):
    """Renders the ensemble analysis with added insights."""
    ensemble_data = report.get("ensemble_analysis")
    if not ensemble_data:
        return

    df = pd.DataFrame(list(ensemble_data.items()), columns=["Layer", "Cosine Similarity"])
    min_similarity_layer = df.loc[df["Cosine Similarity"].idxmin()]

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Cosine Similarity:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Layer:N", sort="-x"),
            tooltip=["Layer", "Cosine Similarity"],
            color=alt.condition(
                alt.datum.Layer == min_similarity_layer["Layer"],
                alt.value("orange"),
                alt.value("steelblue"),
            ),
        )
        .properties(title="Similarity Between Original and Perturbed Activations")
    )
    st.altair_chart(chart, use_container_width=True)

    st.subheader("💡 Insight")
    layer_name = min_similarity_layer["Layer"]
    insight_text = f"The largest divergence occurs at the **'{layer_name}'** layer. "
    if "fc" in layer_name or "classifier" in layer_name or "head" in layer_name:
        insight_text += (
            "This suggests the perturbation altered the model's high-level semantic understanding."
        )
    elif "layer4" in layer_name or "layer3" in layer_name:
        insight_text += (
            "This suggests the attack corrupted complex features like object parts and textures."
        )
    else:
        insight_text += "This suggests the attack altered the model's perception of basic features like edges and colors."  # noqa: E501
    st.info(insight_text)


def render_analysis_tabs(report: dict):  # noqa: C901
    """Renders the detailed analysis sections in a set of tabs."""
    st.header("📈 Detailed Analysis")

    analysis_tabs = []
    if report.get("search_path"):
        analysis_tabs.append("Search Path")
    if report.get("ensemble_analysis"):
        analysis_tabs.append("Ensemble Analysis")
    # Only include Top-K tab if data exists:
    _tk = report.get("top_k_analysis")
    if (report.get("test_type") == "given" and _tk is not None) or (
        isinstance(_tk, list) and len(_tk) > 0
    ):
        analysis_tabs.append("Top-K Overlap")
    if report.get("resolution_impact"):
        analysis_tabs.append("Resolution Impact")
    if report.get("noise_sweep_results"):
        analysis_tabs.append("Noise Sensitivity")
    if report.get("corruption_sweep_results"):
        analysis_tabs.append("CIFAR-10-C Corruptions")
    analysis_tabs.append("Raw Report")

    if len(analysis_tabs) == 1:
        st.info("No detailed analysis data available for this test.")
        st.subheader("Full JSON Report")
        st.json(report)
        return

    tabs = st.tabs(analysis_tabs)
    tab_map = {name: tab for name, tab in zip(analysis_tabs, tabs)}

    if "Search Path" in tab_map:
        with tab_map["Search Path"]:
            st.markdown("**🔍 Search Path Analysis**")
            st.markdown(
                "Shows the adaptive binary search process to find the minimal perturbation that causes the robustness property to fail. Each point represents a tested perturbation level."  # noqa: E501
            )

            df = pd.DataFrame(report["search_path"])
            failure_threshold = report.get("failure_threshold")

            # Generate insights based on search results
            if failure_threshold is not None:
                if failure_threshold < 0.001:
                    st.success(
                        f"🛡️ **High Robustness**: Model is very robust with failure threshold ε = {failure_threshold:.6f}"  # noqa: E501
                    )
                elif failure_threshold < 0.01:
                    st.info(
                        f"🔒 **Moderate Robustness**: Model shows good robustness with failure threshold ε = {failure_threshold:.4f}"  # noqa: E501
                    )
                else:
                    st.warning(
                        f"⚠️ **Low Robustness**: Model is vulnerable with failure threshold ε = {failure_threshold:.4f}"  # noqa: E501
                    )
            else:
                st.info(
                    "🔍 **Search In Progress**: No failure threshold found within the tested range"
                )

            df = pd.DataFrame(report["search_path"])
            if df.empty or "level" not in df.columns:
                st.info("No search path levels to display.")
                st.dataframe(df)
                return
            if "passed" in df.columns and "passed_all" not in df.columns:
                df["passed_all"] = df["passed"]
            df["Status"] = df["passed_all"].apply(lambda x: "✅ Pass" if x else "❌ Fail")

            series_frames = []
            if "confidence" in df.columns:
                series_frames.append(
                    df[["level", "Status"]].assign(series="Confidence", value=df["confidence"])
                )
            if "passed_frac" in df.columns:
                series_frames.append(
                    df[["level", "Status"]].assign(series="Pass Fraction", value=df["passed_frac"])
                )
            if (
                isinstance(report.get("top_k_analysis"), list)
                and len(report.get("top_k_analysis")) > 0
            ):
                df_topk = pd.DataFrame(report["top_k_analysis"])
                dfm = pd.merge(df[["level", "Status"]], df_topk, on="level", how="left")
                if "overlap" in dfm.columns:
                    max_overlap = max(1, int(dfm["overlap"].max()))
                    dfm["value"] = dfm["overlap"] / max_overlap
                    dfm["series"] = "Top-K Overlap (norm)"
                    series_frames.append(dfm[["level", "Status", "series", "value"]])

            if series_frames:
                dfl = pd.concat(series_frames, ignore_index=True)
                # Ensure proper dtypes for Altair
                try:
                    dfl["level"] = pd.to_numeric(dfl["level"], errors="coerce")
                    dfl = dfl.dropna(subset=["level"])
                except (KeyError, ValueError) as e:
                    logger.warning(f"Failed to convert 'level' column to numeric: {e}")
                color_scale = alt.Scale(
                    domain=["Confidence", "Pass Fraction", "Top-K Overlap (norm)"],
                    range=["#1f77b4", "#2ca02c", "orange"],
                )
                chart = (
                    alt.Chart(dfl)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("level:Q", title="Perturbation Level (ε)"),
                        y=alt.Y("value:Q", title="Value (0-1)", scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color("series:N", title="Series", scale=color_scale),
                        strokeDash=alt.condition(
                            alt.datum.Status == "✅ Pass", alt.value([1, 0]), alt.value([4, 2])
                        ),
                        tooltip=["level", "series", "value", "Status"],
                    )
                    .interactive()
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                # Ensure numeric level
                try:
                    df["level"] = pd.to_numeric(df["level"], errors="coerce")
                    df = df.dropna(subset=["level"])
                except (KeyError, ValueError) as e:
                    logger.warning(f"Failed to convert 'level' column to numeric: {e}")
                chart = (
                    alt.Chart(df)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("level:Q", title="Perturbation Level (ε)"),
                        y=alt.Y(
                            "confidence:Q", title="Model Confidence", scale=alt.Scale(domain=[0, 1])
                        ),
                        color=alt.Color(
                            "Status:N",
                            scale=alt.Scale(
                                domain=["✅ Pass", "❌ Fail"], range=["#2ca02c", "#d62728"]
                            ),
                        ),
                        tooltip=["level", "confidence", "prediction", "Status"],
                    )
                    .interactive()
                )
                st.altair_chart(chart, use_container_width=True)

    if "Ensemble Analysis" in tab_map:
        with tab_map["Ensemble Analysis"]:
            st.markdown("**🧠 Ensemble Analysis**")
            st.markdown(
                "Measures how perturbations affect different layers of the neural network using cosine similarity. Higher similarity means the layer's representation is more stable under perturbation."  # noqa: E501
            )
            render_ensemble_analysis(report)

    if "Top-K Overlap" in tab_map:
        with tab_map["Top-K Overlap"]:
            st.markdown("**🎯 Top-K Prediction Overlap**")
            st.markdown(
                "Analyzes how many of the model's top predictions remain consistent between original and perturbed inputs. Higher overlap indicates better prediction stability."  # noqa: E501
            )

            top_k_data = report.get("top_k_analysis")
            k_val = report.get("perturbation_info", {}).get("params", {}).get("top_k", 5)

            if report.get("test_type") == "given":
                st.metric(f"Top-{k_val} Overlap Count", f"{top_k_data:.0f} / {k_val}")

                # Generate insights for fixed tests
                overlap_ratio = top_k_data / k_val if k_val > 0 else 0
                if overlap_ratio == 1.0:
                    st.success(
                        f"🎯 **Perfect Stability**: All top-{k_val} predictions remained consistent"
                    )
                elif overlap_ratio >= 0.8:
                    st.info(
                        f"✅ **Good Stability**: {overlap_ratio:.1%} of top predictions remained consistent"  # noqa: E501
                    )
                elif overlap_ratio >= 0.5:
                    st.warning(
                        f"⚠️ **Moderate Stability**: Only {overlap_ratio:.1%} of top predictions remained consistent"  # noqa: E501
                    )
                else:
                    st.error(
                        f"❌ **Poor Stability**: Only {overlap_ratio:.1%} of top predictions remained consistent"  # noqa: E501
                    )
            else:
                # Guard against empty or invalid series
                if not isinstance(top_k_data, list) or len(top_k_data) == 0:
                    st.info("No Top-K overlap data available for this run.")
                else:
                    df = pd.DataFrame(top_k_data)
                    chart = (
                        alt.Chart(df)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("level:Q", title="Perturbation Level (ε)"),
                            y=alt.Y(
                                "overlap:Q",
                                title="Top-K Overlap Count",
                                scale=alt.Scale(domain=[0, k_val]),
                            ),
                            tooltip=["level", "overlap"],
                        )
                        .properties(
                            title="How Top-K Prediction Overlap Degrades with Attack Strength"
                        )
                    )
                    st.altair_chart(chart, use_container_width=True)

                    # Generate insights for search tests
                    avg_overlap = df["overlap"].mean() if not df.empty else 0
                    if avg_overlap >= k_val * 0.8:
                        st.success(
                            f"🎯 **Stable Predictions**: Average overlap of {avg_overlap:.1f}/{k_val} shows good prediction consistency"  # noqa: E501
                        )
                    elif avg_overlap >= k_val * 0.5:
                        st.warning(
                            f"⚠️ **Degrading Stability**: Average overlap of {avg_overlap:.1f}/{k_val} shows moderate prediction drift"  # noqa: E501
                        )
                    else:
                        st.error(
                            f"❌ **Unstable Predictions**: Average overlap of {avg_overlap:.1f}/{k_val} shows significant prediction changes"  # noqa: E501
                        )

    if "Resolution Impact" in tab_map:
        with tab_map["Resolution Impact"]:
            st.markdown("**📐 Resolution Impact Analysis**")
            st.markdown(
                "Tests how model robustness changes when inputs are resized to different resolutions. This helps understand if robustness is affected by image scale or detail level."  # noqa: E501
            )

            rows = [
                {"Resolution": res, **vals} for res, vals in report["resolution_impact"].items()
            ]
            df = pd.DataFrame(rows)

            base_chart = (
                alt.Chart(df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Resolution:N", sort=None),
                    y=alt.Y("accuracy:Q", title="Accuracy", scale=alt.Scale(domain=[0, 1])),
                )
            )

            tooltip_items = ["Resolution", "accuracy"]
            if "epsilon" in df.columns:
                base_chart = base_chart.encode(
                    color=alt.Color("epsilon:Q", scale=alt.Scale(scheme="blues"), title="ε")
                )
                tooltip_items.append("epsilon")

            final_chart = base_chart.encode(tooltip=tooltip_items).properties(
                title="Accuracy vs Resolution"
            )
            st.altair_chart(final_chart, use_container_width=True)

            # Generate insights
            if not df.empty and "accuracy" in df.columns:
                max_acc = df["accuracy"].max()
                min_acc = df["accuracy"].min()
                acc_range = max_acc - min_acc
                if acc_range < 0.1:
                    st.success(
                        f"🔒 **Resolution Invariant**: Accuracy varies by only {acc_range:.1%} across resolutions"  # noqa: E501
                    )
                elif acc_range < 0.3:
                    st.info(
                        f"📊 **Moderate Variation**: Accuracy varies by {acc_range:.1%} across resolutions"  # noqa: E501
                    )
                else:
                    st.warning(
                        f"⚠️ **High Variation**: Accuracy varies by {acc_range:.1%} - model is sensitive to resolution"  # noqa: E501
                    )

            st.dataframe(df)

    if "Noise Sensitivity" in tab_map:
        with tab_map["Noise Sensitivity"]:
            st.markdown("**🌪️ Noise Sensitivity Analysis**")
            st.markdown(
                "Evaluates how model performance degrades under increasing levels of Gaussian noise. This simulates real-world conditions like sensor noise or image compression artifacts."  # noqa: E501
            )

            df = pd.DataFrame(report["noise_sweep_results"])
            chart = (
                alt.Chart(df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("level:Q", title="Noise Level (Std. Dev)"),
                    y=alt.Y("accuracy:Q", title="Accuracy", scale=alt.Scale(domain=[0, 1])),
                    tooltip=["level", "accuracy"],
                )
                .properties(title="Robustness vs. Gaussian Noise")
            )
            st.altair_chart(chart, use_container_width=True)

            # Generate insights
            if not df.empty and "accuracy" in df.columns:
                clean_acc = df["accuracy"].iloc[0] if len(df) > 0 else 1.0
                final_acc = df["accuracy"].iloc[-1] if len(df) > 0 else 0.0
                acc_drop = clean_acc - final_acc
                if acc_drop < 0.1:
                    st.success(
                        f"🛡️ **Noise Robust**: Only {acc_drop:.1%} accuracy drop under maximum noise"  # noqa: E501
                    )
                elif acc_drop < 0.3:
                    st.info(
                        f"🔒 **Moderate Sensitivity**: {acc_drop:.1%} accuracy drop under maximum noise"  # noqa: E501
                    )
                else:
                    st.warning(
                        f"⚠️ **High Sensitivity**: {acc_drop:.1%} accuracy drop - model is vulnerable to noise"  # noqa: E501
                    )

    if "CIFAR-10-C Corruptions" in tab_map:
        with tab_map["CIFAR-10-C Corruptions"]:
            st.markdown("**🌊 CIFAR-10-C Corruption Analysis**")
            st.markdown(
                "Tests robustness against common image corruptions (blur, noise, weather effects, etc.) at different severity levels. This evaluates real-world robustness beyond adversarial attacks."  # noqa: E501
            )

            corr = report["corruption_sweep_results"]
            rows = []
            for name, series in corr.items():
                for item in series:
                    rows.append({"corruption": name, **item})
            if rows:
                df = pd.DataFrame(rows)
                chart = (
                    alt.Chart(df)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("severity:O", title="Severity (1–5)"),
                        y=alt.Y("accuracy:Q", title="Accuracy", scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color("corruption:N", title="Corruption"),
                        tooltip=["corruption", "severity", "accuracy"],
                    )
                    .properties(title="Accuracy vs. CIFAR-10-C Corruption Severity")
                )
                st.altair_chart(chart, use_container_width=True)

                # Generate insights
                if "accuracy" in df.columns:
                    avg_acc_sev5 = (
                        df[df["severity"] == 5]["accuracy"].mean()
                        if not df[df["severity"] == 5].empty
                        else 0.0
                    )
                    if avg_acc_sev5 > 0.7:
                        st.success(
                            f"🛡️ **Corruption Robust**: Average {avg_acc_sev5:.1%} accuracy at maximum severity"  # noqa: E501
                        )
                    elif avg_acc_sev5 > 0.4:
                        st.info(
                            f"🔒 **Moderate Robustness**: Average {avg_acc_sev5:.1%} accuracy at maximum severity"  # noqa: E501
                        )
                    else:
                        st.warning(
                            f"⚠️ **Corruption Vulnerable**: Only {avg_acc_sev5:.1%} accuracy at maximum severity"  # noqa: E501
                        )

                st.dataframe(df)

    if "Raw Report" in tab_map:
        with tab_map["Raw Report"]:
            st.markdown("**📋 Raw JSON Report**")
            st.markdown(
                "Complete test results in JSON format. Useful for debugging, custom analysis, or integration with other tools."  # noqa: E501
            )
            st.json(report)


def render_all_reports(results: dict):  # noqa: C901
    """Iterates through all test results and renders them in expanders."""
    for test_name, report in results.items():
        with st.expander(f"Test: `{test_name}`", expanded=True):
            render_key_metrics(report)
            st.markdown("---")

            render_strategies_section(report)
            st.markdown("---")

            st.header("🖼️ Visual Comparison")
            # Brief dynamic note about resolutions instead of hardcoded dataset
            o = report.get("original_image") or {}
            p = report.get("perturbed_image") or {}
            try:
                ob = _get_image_bytes(o) if o else None
                pb = _get_image_bytes(p) if p else None
                ow = oh = pw = ph = None
                if ob:
                    oi = Image.open(io.BytesIO(ob))
                    ow, oh = oi.size
                if pb:
                    pi = Image.open(io.BytesIO(pb))
                    pw, ph = pi.size
                parts = []
                if ow and oh:
                    parts.append(f"Original: {ow}×{oh}px")
                if pw and ph:
                    parts.append(f"Perturbed: {pw}×{ph}px")
                if parts:
                    st.info("; ".join(parts) + ". Small images may look blocky when enlarged.")
            except (OSError, ValueError, AttributeError) as e:
                logger.debug(f"Failed to extract image size info: {e}")
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
                    st.subheader("Residual (Size of change)")
                    # Render residual panel with same fixed width, pixelated scaling
                    try:
                        rimg = report["residual_image"]
                        rbytes = _get_image_bytes(rimg)
                        rcap = rimg.get("caption", "Residual")
                        if rbytes:
                            rb64 = base64.b64encode(rbytes).decode("utf-8")
                            html = f'<img class="pixelated" src="data:image/png;base64,{rb64}" width="200"/>'  # noqa: E501
                            st.markdown(html, unsafe_allow_html=True)
                            st.caption(rcap)

                        # Display residual metrics prominently
                        if report.get("residual_metrics"):
                            metrics = report["residual_metrics"]
                            st.markdown("**Residual Metrics:**")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(
                                    "L∞ Norm",
                                    f"{metrics.get('linf_norm', 0):.4f}",
                                    help="Maximum absolute pixel difference",
                                )

                    except (KeyError, TypeError, ValueError) as e:
                        logger.warning(f"Failed to render residual panel: {e}")
                        st.warning("Residual panel could not be rendered.")

            st.markdown("---")
            render_analysis_tabs(report)


# ============================================================================
# NEW: EXECUTIVE SUMMARY & FAILURE ANALYSIS FUNCTIONS
# ============================================================================


def render_executive_summary(report: dict):
    """Renders the executive summary with interpreted metrics and action signals."""
    st.header("📊 Executive Summary")

    col1, col2, col3 = st.columns(3)

    # Robustness score with interpretation
    with col1:
        test_type = report.get("test_type", "unknown")

        if test_type == "given":
            accuracy = report.get("robust_accuracy")
            if accuracy is not None:
                st.metric(
                    "Robust Accuracy",
                    f"{accuracy:.1%}",
                    help="Fraction of samples maintaining correct prediction",
                )
                # Interpretation
                if accuracy > 0.8:
                    st.success("✅ **Strong robustness** - Ready for evaluation")
                elif accuracy > 0.6:
                    st.warning("⚠️ **Moderate robustness** - Review findings before deployment")
                else:
                    st.error("❌ **Weak robustness** - Requires significant improvement")

        elif test_type == "search":
            threshold = report.get("failure_threshold")
            if threshold is not None:
                st.metric(
                    "Failure Threshold (ε)",
                    f"{threshold:.5f}",
                    help="Minimum perturbation causing property failure",
                )
                # Interpretation
                if threshold < 0.001:
                    st.success("✅ **High robustness** - Very resistant to perturbations")
                elif threshold < 0.01:
                    st.info("🔒 **Moderate robustness** - Good overall resistance")
                else:
                    st.warning("⚠️ **Low robustness** - Vulnerable to small perturbations")

    # Critical failures
    with col2:
        per_sample = report.get("per_sample_metrics", [])
        if per_sample:
            critical_failures = sum(
                1 for s in per_sample if s.get("confidence_drop", 0) > 0.5
            )
            st.metric("Critical Failures", critical_failures, help=">50% confidence drop")
            if critical_failures > 0:
                st.error(
                    f"⚠️ {critical_failures} images show severe robustness issues"
                )
            else:
                st.success("✅ No severe failures detected")
        else:
            st.metric("Critical Failures", "—")

    # Most vulnerable class
    with col3:
        per_sample = report.get("per_sample_metrics", [])
        if per_sample:
            class_failures = Counter()
            for s in per_sample:
                if not s.get("passed", True):
                    cls = s.get("clean_label", "Unknown")
                    class_failures[cls] += 1

            if class_failures:
                most_vulnerable = class_failures.most_common(1)[0]
                st.metric(
                    "Most Vulnerable Class",
                    most_vulnerable[0],
                    f"{most_vulnerable[1]} failures",
                )
                st.warning(f"Focus efforts on improving {most_vulnerable[0]} robustness")
            else:
                st.metric("Most Vulnerable Class", "—")


def extract_failures_from_report(report: dict) -> List[Dict[str, Any]]:
    """Extract and enrich failure data from report for analysis."""
    failures = []
    per_sample = report.get("per_sample_metrics", [])

    for i, sample in enumerate(per_sample):
        if not sample.get("passed", True):
            failure = {
                "index": i,
                "original_pred": sample.get("clean_label", "?"),
                "perturbed_pred": sample.get("pert_label", "?"),
                "confidence_drop": sample.get("confidence_drop", 0),
                "strategy": report.get("perturbation_info", {}).get("name", "Unknown"),
                "topk_overlap": sample.get("topk_overlap"),
                "original_image": None,  # Would be populated if available
                "perturbed_image": None,
            }
            failures.append(failure)

    return sorted(failures, key=lambda f: f["confidence_drop"], reverse=True)


def render_failure_triage(report: dict):
    """Renders interactive failure triage section with ranking and filtering."""
    st.header("🔴 Failures to Investigate")

    failures = extract_failures_from_report(report)

    if not failures:
        st.success("✅ No failures detected! Model is robust to this perturbation.")
        return

    st.write(f"Found **{len(failures)}** failures out of {len(report.get('per_sample_metrics', []))} samples")

    # Create tabs for different triage views
    tab1, tab2, tab3 = st.tabs(["By Severity", "By Class", "By Pattern"])

    with tab1:  # Severity-based triage
        st.subheader("Ranked by Confidence Drop")

        # Filter by severity threshold
        severity_threshold = st.slider(
            "Minimum confidence drop to show",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
        )

        filtered_failures = [f for f in failures if f["confidence_drop"] >= severity_threshold]

        st.write(f"Showing {len(filtered_failures)}/{len(failures)} failures")

        # Create severity ranking table
        table_data = []
        for i, failure in enumerate(filtered_failures[:15]):
            table_data.append({
                "Rank": i + 1,
                "Original": failure["original_pred"],
                "Predicted": failure["perturbed_pred"],
                "Conf. Drop": f"{failure['confidence_drop']:.1%}",
                "Severity": (
                    "🔴 Critical"
                    if failure["confidence_drop"] > 0.7
                    else "🟠 High"
                    if failure["confidence_drop"] > 0.5
                    else "🟡 Medium"
                ),
            })

        if table_data:
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Show top 3 with explanation
            st.subheader("Top Critical Cases")
            for failure in filtered_failures[:3]:
                with st.expander(
                    f"Case {failure['index']}: {failure['original_pred']} → {failure['perturbed_pred']} "
                    f"(-{failure['confidence_drop']:.1%})"
                ):
                    col1, col2 = st.columns([2, 2])
                    with col1:
                        st.markdown("**Why it failed:**")
                        st.info(
                            "• Confidence severely degraded\n"
                            "• Model became highly uncertain\n"
                            "• Different class predicted at high confidence"
                        )
                    with col2:
                        st.markdown("**Suggested actions:**")
                        st.write(
                            "✏️ Add more training samples of this class\n"
                            "🎨 Use stronger augmentation\n"
                            "🔍 Investigate perturbation parameters"
                        )

    with tab2:  # Class-based triage
        st.subheader("Failures Grouped by Class")

        class_failures = defaultdict(list)
        for failure in failures:
            cls = failure["original_pred"]
            class_failures[cls].append(failure)

        # Create class failure summary
        class_summary = []
        for cls, cls_failures in sorted(
            class_failures.items(), key=lambda x: len(x[1]), reverse=True
        ):
            avg_drop = np.mean([f["confidence_drop"] for f in cls_failures])
            class_summary.append({
                "Class": cls,
                "Failures": len(cls_failures),
                "Avg. Confidence Drop": f"{avg_drop:.1%}",
                "Severity": (
                    "🔴 Critical"
                    if avg_drop > 0.7
                    else "🟠 High"
                    if avg_drop > 0.5
                    else "🟡 Medium"
                ),
            })

        df_class = pd.DataFrame(class_summary)
        st.dataframe(df_class, use_container_width=True, hide_index=True)

        # Drill into most vulnerable class
        if class_summary:
            most_vulnerable = class_summary[0]
            st.subheader(f"Details: {most_vulnerable['Class']} ({most_vulnerable['Failures']} failures)")

            cls_failures = class_failures[most_vulnerable["Class"]]
            st.write(f"Average confidence drop: **{most_vulnerable['Avg. Confidence Drop']}**")

            st.info(
                f"💡 **Recommendation:** This class is your priority. "
                f"Consider collecting more training data or applying stronger augmentation."
            )

    with tab3:  # Pattern-based triage
        st.subheader("Failure Patterns")

        # Analyze patterns
        severe_failures = [f for f in failures if f["confidence_drop"] > 0.7]
        moderate_failures = [f for f in failures if 0.5 <= f["confidence_drop"] <= 0.7]
        mild_failures = [f for f in failures if f["confidence_drop"] < 0.5]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("🔴 Critical", len(severe_failures), help=">70% confidence drop")

        with col2:
            st.metric("🟠 High", len(moderate_failures), help="50-70% drop")

        with col3:
            st.metric("🟡 Medium", len(mild_failures), help="<50% drop")

        st.markdown("---")

        # Pattern breakdown
        st.write("**Failure Distribution:**")

        if severe_failures:
            with st.expander("🔴 Critical Failures (70%+ drop)"):
                st.write(f"Count: {len(severe_failures)} cases")
                for f in severe_failures[:5]:
                    st.write(
                        f"• {f['original_pred']} → {f['perturbed_pred']} "
                        f"({f['confidence_drop']:.1%})"
                    )

        if moderate_failures:
            with st.expander("🟠 High Priority (50-70% drop)"):
                st.write(f"Count: {len(moderate_failures)} cases")
                for f in moderate_failures[:5]:
                    st.write(
                        f"• {f['original_pred']} → {f['perturbed_pred']} "
                        f"({f['confidence_drop']:.1%})"
                    )

        if mild_failures:
            with st.expander("🟡 Medium Priority (<50% drop)"):
                st.write(f"Count: {len(mild_failures)} cases")
                for f in mild_failures[:5]:
                    st.write(
                        f"• {f['original_pred']} → {f['perturbed_pred']} "
                        f"({f['confidence_drop']:.1%})"
                    )


def render_root_cause_analysis(report: dict):
    """Renders root cause analysis with patterns and insights."""
    st.header("🔍 Root Cause Analysis")

    per_sample = report.get("per_sample_metrics", [])
    if not per_sample:
        st.info("No per-sample data available for analysis")
        return

    col1, col2 = st.columns(2)

    # Pass/Fail statistics
    with col1:
        total = len(per_sample)
        passed = sum(1 for s in per_sample if s.get("passed", True))
        failed = total - passed

        st.metric("Samples Passed", f"{passed}/{total}", f"{passed/total:.1%}")
        st.metric("Samples Failed", f"{failed}/{total}", f"{failed/total:.1%}")

    # Confidence analysis
    with col2:
        confidence_drops = [s.get("confidence_drop", 0) for s in per_sample]
        avg_drop = np.mean(confidence_drops)
        max_drop = np.max(confidence_drops)

        st.metric("Avg Confidence Drop", f"{avg_drop:.1%}")
        st.metric("Max Confidence Drop", f"{max_drop:.1%}")

    st.markdown("---")

    # Confidence drop distribution
    st.subheader("Confidence Drop Distribution")

    fig = alt.Chart(
        pd.DataFrame({"Confidence Drop": confidence_drops})
    ).mark_histogram(bins=20).encode(
        x=alt.X("Confidence Drop:Q", title="Confidence Drop"),
        y=alt.Y("count():Q", title="Number of Samples"),
    ).properties(
        height=300
    )

    st.altair_chart(fig, use_container_width=True)

    # Insight box
    if max_drop > 0.7:
        st.warning(
            f"⚠️ **High Degradation:** Some samples show >70% confidence drop, "
            f"indicating severe vulnerability."
        )
    elif avg_drop > 0.5:
        st.info(
            f"📊 **Moderate Degradation:** Average {avg_drop:.1%} drop suggests "
            f"meaningful but not catastrophic robustness loss."
        )
    else:
        st.success(
            f"✅ **Low Degradation:** Average {avg_drop:.1%} drop indicates good robustness."
        )


def render_search_path_insights(report: dict):
    """Renders insights about the adaptive search process."""
    st.header("🎯 Adaptive Search Analysis")

    search_path = report.get("search_path")
    if not search_path:
        st.info("No search path available for this test")
        return

    failure_threshold = report.get("failure_threshold")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Search Steps",
            len(search_path),
            help="Number of adaptive search iterations",
        )

    with col2:
        # Rough estimate of grid search cost
        if failure_threshold is not None:
            estimated_steps = estimate_grid_search_equivalent(failure_threshold)
            st.metric(
                "Grid Search Est.",
                estimated_steps,
                help="Estimated steps for equivalent grid search",
            )

    with col3:
        if failure_threshold is not None and len(search_path) > 0:
            efficiency = estimated_steps / len(search_path)
            st.metric(f"Efficiency Gain", f"{efficiency:.1f}×")

    st.markdown("---")

    # Search convergence visualization
    st.subheader("Search Convergence")

    df_search = pd.DataFrame(search_path)
    if not df_search.empty and "level" in df_search.columns:
        try:
            df_search["level"] = pd.to_numeric(df_search["level"], errors="coerce")

            # Create visualization
            if "passed_all" in df_search.columns or "passed" in df_search.columns:
                pass_col = "passed_all" if "passed_all" in df_search.columns else "passed"
                df_search["Status"] = df_search[pass_col].apply(
                    lambda x: "✅ Pass" if x else "❌ Fail"
                )

                fig = alt.Chart(df_search).mark_point(size=100).encode(
                    x=alt.X("level:Q", title="Perturbation Level (ε)"),
                    y=alt.Y("confidence:Q", title="Model Confidence", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color(
                        "Status:N",
                        scale=alt.Scale(
                            domain=["✅ Pass", "❌ Fail"],
                            range=["#2ca02c", "#d62728"],
                        ),
                    ),
                    tooltip=["level", "confidence", "Status"],
                ).interactive()

                st.altair_chart(fig, use_container_width=True)

                # Insight
                if failure_threshold is not None:
                    st.info(
                        f"💡 **Convergence:** Found failure threshold at ε = {failure_threshold:.5f} "
                        f"in {len(search_path)} steps using adaptive binary search."
                    )

        except Exception as e:
            logger.warning(f"Failed to visualize search path: {e}")
            st.dataframe(df_search)


def render_actionable_recommendations(report: dict):
    """Renders prioritized, actionable recommendations based on findings."""
    st.header("🔧 Recommended Actions")

    recommendations = generate_recommendations(report)

    if not recommendations:
        st.success("✅ No critical issues to address")
        return

    # Sort by priority (High > Medium > Low)
    recommendations = sorted(
        recommendations,
        key=lambda r: (0 if "High" in r["priority"] else 1 if "Medium" in r["priority"] else 2),
    )

    for i, rec in enumerate(recommendations, 1):
        with st.expander(
            f"{rec['priority']} Priority {i}: {rec['title']}", expanded=(i == 1)
        ):
            st.write(rec["description"])

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Evidence")
                st.write(rec["evidence"])

            with col2:
                st.subheader("Expected Impact")
                st.write(rec["impact"])

            st.subheader("How to Implement")
            st.code(rec["code_example"], language="python")


def generate_recommendations(report: dict) -> List[Dict[str, str]]:
    """Generate prioritized recommendations based on report findings."""
    recommendations = []
    per_sample = report.get("per_sample_metrics", [])

    if not per_sample:
        return recommendations

    # Analyze failures
    failures = [s for s in per_sample if not s.get("passed", True)]
    failure_rate = len(failures) / len(per_sample) if per_sample else 0

    # Rec 1: High failure rate
    if failure_rate > 0.5:
        recommendations.append({
            "priority": "🔴 High",
            "title": "Overall Robustness is Low",
            "description": (
                f"Your model shows a {failure_rate:.1%} failure rate under the applied perturbations. "
                f"This indicates fundamental fragility that requires comprehensive mitigation."
            ),
            "evidence": f"{len(failures)}/{len(per_sample)} samples failed",
            "impact": (
                "Improving robustness typically requires:\n"
                "• Augmented training with perturbations\n"
                "• Data augmentation strategies\n"
                "• Ensemble methods"
            ),
            "code_example": (
                "from torchvision import transforms\n\n"
                "train_augmentation = transforms.Compose([\n"
                "    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),\n"
                "    transforms.ColorJitter(brightness=0.2, contrast=0.2),\n"
                "    transforms.GaussianBlur(kernel_size=3),\n"
                "])\n\n"
                "# Retrain model with augmented data"
            ),
        })

    # Rec 2: Class imbalance in failures
    class_failures = defaultdict(int)
    for failure in failures:
        cls = failure.get("clean_label", "Unknown")
        class_failures[cls] += 1

    if class_failures:
        most_vulnerable = max(class_failures.items(), key=lambda x: x[1])
        cls_name, cls_count = most_vulnerable
        cls_rate = cls_count / len([s for s in per_sample if s.get("clean_label") == cls_name])

        if cls_rate > 0.4:
            recommendations.append({
                "priority": "⚠️ Medium",
                "title": f'Improve "{cls_name}" Robustness',
                "description": (
                    f'"{cls_name}" has the highest failure rate ({cls_rate:.1%}). '
                    f"This class is disproportionately vulnerable to the perturbation."
                ),
                "evidence": f"{cls_count} failures out of class {cls_name}",
                "impact": (
                    "Class-specific improvements:\n"
                    "• Collect more examples of this class\n"
                    "• Apply stronger augmentation to this class\n"
                    "• Use class-specific regularization"
                ),
                "code_example": (
                    f'class_specific_data = [\n'
                    f'    x for x, y in train_data if y == "{cls_name}"\n'
                    f']\n\n'
                    f"# Apply stronger augmentation to this class\n"
                    f"augmented = [augment_heavily(x) for x in class_specific_data]"
                ),
            })

    # Rec 3: Severe confidence drops
    severe_drops = [f for f in failures if f.get("confidence_drop", 0) > 0.7]
    if len(severe_drops) > 2:
        recommendations.append({
            "priority": "🔴 High",
            "title": "Address Severe Confidence Degradation",
            "description": (
                f"{len(severe_drops)} samples show >70% confidence drop, indicating "
                f"the model is becoming extremely uncertain under perturbation."
            ),
            "evidence": f"{len(severe_drops)}/{len(failures)} failures are severe",
            "impact": (
                "Severe degradation can be addressed by:\n"
                "• Confidence calibration techniques\n"
                "• Ensemble averaging\n"
                "• Temperature scaling\n"
                "• Training with uncertainty awareness"
            ),
            "code_example": (
                "# Temperature scaling for calibration\n"
                "temperature = 1.5\n"
                "calibrated_probs = torch.softmax(logits / temperature, dim=1)"
            ),
        })

    return recommendations


def estimate_grid_search_equivalent(failure_threshold: float, grid_points: int = 100) -> int:
    """Estimate how many grid search steps would be needed."""
    # This is a simplified estimate - assumes binary precision refinement
    if failure_threshold == 0:
        return grid_points

    # In a grid search, we'd need to evaluate all points up to failure threshold
    # with some granularity
    granularity = 1 / grid_points
    estimated_steps = int(failure_threshold / granularity)

    return max(estimated_steps, grid_points)
