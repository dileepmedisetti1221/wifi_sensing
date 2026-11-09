import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import re
import joblib
from datetime import datetime
from collections import deque
from streamlit_autorefresh import st_autorefresh
import io
import zipfile
import os

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def get_feature_names():
    """Feature column names used by the DecisionTree model.
    Mirrors the ``FEATURES`` list from ``train_model.py``.
    """
    return ["mean_rssi", "variance", "std_rssi", "min_rssi", "max_rssi", "range_rssi"]

@st.cache_data(show_spinner=False)
def load_historical_data():
    """Load the synthetic feature dataset.
    Returns a DataFrame and a pre‑computed per‑class summary.
    """
    df = pd.read_csv(os.path.join("data", "features.csv"))
    # Per‑class aggregate statistics for the six features
    summary = (
        df.groupby("label")[get_feature_names()]
        .agg(["mean", "var", "std"])  # mean, variance, std per class
        .reset_index()
    )
    # Flatten multi‑index columns for easier display
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.values]
    return df, summary

@st.cache_resource(show_spinner=False)
def load_model_and_features():
    """Load the saved model and retrieve the feature order.
    Returns ``model`` and ``features`` (list of column names in the order the model expects).
    """
    model_path = os.path.join("models", "occupancy_model.joblib")
    model_data = joblib.load(model_path)
    model = model_data["model"]
    features = model_data.get("features", get_feature_names())
    return model, features

def get_wifi_signal():
    """Read the current Wi‑Fi signal strength via ``netsh``.
    Returns ``(percentage, estimated_rssi)`` or ``(None, None)`` on failure.
    """
    try:
        result = subprocess.run(
            ["netsh", "wlan", "show", "interfaces"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        match = re.search(r"Signal\s*:\s*(\d+)\s*%", result.stdout, re.IGNORECASE)
        if match:
            percentage = int(match.group(1))
            # Rough conversion from % to dBm (same as in the logger)
            estimated_rssi = -100 + percentage * 0.5
            return percentage, estimated_rssi
        return None, None
    except Exception:
        return None, None

# ---------------------------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Wi‑Fi Occupancy Dashboard", page_icon="📶", layout="wide")
st.title("📶 Wi‑Fi Sensing and Occupancy Detection")
st.caption("Device‑free indoor occupancy classification demonstration (synthetic data)")

# ---------------------------------------------------------------------------
# Load model & data once
# ---------------------------------------------------------------------------
model, model_features = load_model_and_features()
full_df, class_summary = load_historical_data()

# ---------------------------------------------------------------------------
# UI layout – tabs for each major functionality
# ---------------------------------------------------------------------------
live_tab, hist_tab, eval_tab, report_tab = st.tabs([
    "Live Occupancy",
    "Historical Analysis",
    "Model Evaluation",
    "Download Report",
])

# ---------------------------------------------------------------------------
# Live Occupancy Classification (existing functionality)
# ---------------------------------------------------------------------------
with live_tab:
    # Refresh every 2 seconds – keeps the UI responsive
    st_autorefresh(interval=2000, key="wifi_refresh")
    # Session state buffers
    if "readings" not in st.session_state:
        st.session_state.readings = deque(maxlen=20)
    if "history" not in st.session_state:
        st.session_state.history = deque(maxlen=60)

    percentage, rssi = get_wifi_signal()
    if percentage is not None:
        now = datetime.now()
        st.session_state.readings.append(rssi)
        st.session_state.history.append({"Time": now.strftime("%H:%M:%S"), "RSSI": rssi, "Signal": percentage})
        st.success("Wi‑Fi connection detected")
        col1, col2, col3 = st.columns(3)
        col1.metric("Signal Strength", f"{percentage}%")
        col2.metric("Estimated RSSI", f"{rssi:.1f} dBm")
        if len(st.session_state.readings) >= 20:
            vals = np.array(st.session_state.readings)
            # Build a one‑row DataFrame with the exact feature order the model expects
            row = pd.DataFrame([
                {
                    "mean_rssi": np.mean(vals),
                    "variance": np.var(vals, ddof=1),
                    "std_rssi": np.std(vals, ddof=1),
                    "min_rssi": np.min(vals),
                    "max_rssi": np.max(vals),
                    "range_rssi": np.max(vals) - np.min(vals),
                }
            ])[model_features]
            pred = model.predict(row)[0]
            col3.metric("Predicted Occupancy", pred)
            st.subheader("Occupancy Classification")
            if pred == "Empty":
                st.info("Predicted state: Empty")
            elif pred == "Moderate":
                st.warning("Predicted state: Moderate")
            else:
                st.error("Predicted state: Crowded")
        else:
            col3.metric("Predicted Occupancy", "Collecting data")
            st.info(f"Collecting readings: {len(st.session_state.readings)}/20")
        # Historical plot for the live session
        history_df = pd.DataFrame(st.session_state.history)
        if not history_df.empty:
            st.line_chart(history_df.set_index("Time")["RSSI"])
        st.subheader("Recent Readings")
        st.dataframe(history_df.tail(10), use_container_width=True)
    else:
        st.error("Wi‑Fi signal unavailable. Connect to a Wi‑Fi network.")

    st.caption(
        "Research prototype: RSSI is approximate, and predictions are not validated for real campus occupancy."
    )

# ---------------------------------------------------------------------------
# Historical Signal Analysis
# ---------------------------------------------------------------------------
with hist_tab:
    st.subheader("Class‑wise Feature Statistics (Synthetic Data)")
    st.dataframe(class_summary, use_container_width=True)

    st.subheader("RSSI Distribution per Occupancy State")
    # Box‑plot using Plotly (works out‑of‑the‑box with Streamlit)
    import plotly.express as px

    fig = px.box(
        full_df,
        x="label",
        y="mean_rssi",
        points="all",
        title="Mean RSSI per Occupancy Class",
        labels={"label": "Occupancy", "mean_rssi": "Mean RSSI (dBm)"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # If a Moderate class is present, show a comparison table
    if "Moderate" in full_df["label"].unique():
        st.subheader("Mean RSSI Comparison")
        comp = (
            full_df.groupby("label")["mean_rssi"]
            .agg(["mean", "std"])  # show mean ± std
            .reset_index()
        )
        st.table(comp)
    else:
        st.info("No Moderate synthetic recordings are available in the current dataset.")

# ---------------------------------------------------------------------------
# Model Evaluation – load results, show confusion matrix and metrics
# ---------------------------------------------------------------------------
with eval_tab:
    # Confusion matrix image (generated by train_model.py)
    cm_path = os.path.join("results", "confusion_matrix.png")
    if os.path.exists(cm_path):
        st.image(cm_path, caption="Confusion Matrix", use_column_width=True)
    else:
        st.warning("Confusion matrix image not found. Run `train_model.py` to generate it.")

    # Load test predictions and compute detailed metrics
    pred_path = os.path.join("results", "test_predictions.csv")
    if os.path.exists(pred_path):
        preds_df = pd.read_csv(pred_path)
        from sklearn.metrics import classification_report, accuracy_score

        y_true = preds_df["label"]
        y_pred = preds_df["predicted_label"]
        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose().round(3)
        st.subheader("Classification Report")
        st.metric("Accuracy", f"{acc:.3%}")
        st.dataframe(report_df, use_container_width=True)
    else:
        st.warning("Test predictions CSV not found. Run `train_model.py` to generate it.")

# ---------------------------------------------------------------------------
# Downloadable Experiment Report
# ---------------------------------------------------------------------------
with report_tab:
    st.info(
        "Package the key artefacts (confusion matrix, test predictions, and class summary) "
        "into a ZIP file for easy sharing."
    )
    # Prepare in‑memory ZIP
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add confusion matrix image if present
        if os.path.exists(cm_path):
            zipf.write(cm_path, arcname="confusion_matrix.png")
        # Add test predictions CSV
        if os.path.exists(pred_path):
            zipf.write(pred_path, arcname="test_predictions.csv")
        # Add class summary CSV (generated on the fly)
        summary_csv = class_summary.to_csv(index=False)
        zipf.writestr("class_summary.csv", summary_csv)
        # Add original feature dataset for completeness
        feature_csv_path = os.path.join("data", "features.csv")
        if os.path.exists(feature_csv_path):
            zipf.write(feature_csv_path, arcname="features.csv")
    buffer.seek(0)
    st.download_button(
        label="Download Experiment Report (ZIP)",
        data=buffer,
        file_name="wifi_occupancy_report.zip",
        mime="application/zip",
    )
