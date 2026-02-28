"""
Streamlit Monitoring & Explainability Dashboard (offline / stub-friendly).

Run:
  streamlit run dashboard.py

This dashboard is intentionally built with stub data sources so you can plug it
into real logs/metrics stores later (see TODOs).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
import streamlit as st
import yaml


# -----------------------------
# Data sources (MLflow + real predictions)
# -----------------------------

@st.cache_data
def get_performance_history(days: int = 30) -> pd.DataFrame:
    """
    Return recent model performance metrics over time from MLflow.

    Falls back to synthetic data if no MLflow runs are found so the dashboard
    still renders even before the first training.
    """
    days = int(days)

    try:
        client = mlflow.tracking.MlflowClient()
        # Try to find an experiment for this project; fall back to default.
        exp = client.get_experiment_by_name("NetworkSecurity") or client.get_experiment_by_name("Default")
        if exp is None:
            raise RuntimeError("No MLflow experiment found.")

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["attributes.start_time ASC"],
        )
        if not runs:
            raise RuntimeError("No MLflow runs found.")

        records = []
        for r in runs:
            m = r.data.metrics
            if "f1_score" not in m:
                continue
            ts = dt.datetime.fromtimestamp(r.info.start_time / 1000.0)
            records.append(
                {
                    "date": ts.date(),
                    "f1_score": float(m.get("f1_score", 0.0)),
                    "precision": float(m.get("precision", 0.0)),
                    "recall": float(m.get("recall_score", 0.0)),
                }
            )

        if records:
            df = pd.DataFrame(records)
            df = df.groupby("date", as_index=False).mean()
            # Keep only the last `days` days.
            df = df.sort_values("date").tail(days)
            return df

    except Exception:
        # Fall back to synthetic history below.
        pass

    end = dt.date.today()
    dates = pd.date_range(end=end, periods=days, freq="D")
    rng = np.random.default_rng(42)
    base = 0.90 + 0.02 * rng.standard_normal(size=days)
    precision = np.clip(base + 0.01 * rng.standard_normal(size=days), 0.0, 1.0)
    recall = np.clip(base + 0.01 * rng.standard_normal(size=days), 0.0, 1.0)
    f1 = np.clip((2 * precision * recall) / (precision + recall + 1e-12), 0.0, 1.0)

    return pd.DataFrame(
        {
            "date": dates,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
        }
    )


@st.cache_data
def get_recent_predictions(limit: int = 100) -> pd.DataFrame:
    """
    Return a table of the most recent predictions from the FastAPI output CSV.

    - Reads `prediction_output/output.csv` that is written by `/predict` in `app.py`.
    - If that file is missing, falls back to synthetic demo data.
    """
    limit = int(limit)
    csv_path = Path("prediction_output") / "output.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # Use the file modification time as a crude timestamp for now.
        ts = dt.datetime.fromtimestamp(csv_path.stat().st_mtime)
        df = df.copy()
        df["timestamp"] = ts
        # Optionally keep only some key columns.
        keep_cols = [c for c in df.columns if c in {"URL_Length", "having_IP_Address", "SSLfinal_State", "Result", "predicted_column"}]
        other_cols = [c for c in df.columns if c not in keep_cols + ["timestamp"]]
        ordered_cols = ["timestamp"] + keep_cols + other_cols
        df = df[ordered_cols]
        return df.tail(limit)

    # Fallback: synthetic predictions.
    rng = np.random.default_rng(7)
    now = dt.datetime.now()

    df = pd.DataFrame(
        {
            "timestamp": [now - dt.timedelta(minutes=5 * i) for i in range(limit)][::-1],
            "URL_Length": rng.integers(10, 100, size=limit),
            "having_IP_Address": rng.integers(0, 2, size=limit),
            "SSLfinal_State": rng.integers(0, 2, size=limit),
            "model_version": ["candidate"] * limit,
            "prediction": rng.integers(0, 2, size=limit),
            "confidence": np.round(rng.uniform(0.55, 0.99, size=limit), 3),
        }
    )
    return df


# -----------------------------
# Artifact loading helpers
# -----------------------------

def _find_latest_artifact_dir(artifacts_root: str = "Artifacts") -> Optional[Path]:
    root = Path(artifacts_root)
    if not root.exists():
        return None
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        return None
    # Artifact dirs are timestamped like 07_04_2025_12_17_18; lexicographic sort works.
    return sorted(dirs)[-1]


def _load_latest_drift_report() -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
    """
    Load KS-test drift report produced by `DataValidation` (YAML).

    Expected schema (per feature):
      { "p_value": float, "drift_status": bool }
    """
    candidate_paths = []
    latest = _find_latest_artifact_dir()
    if latest is not None:
        candidate_paths.append(latest / "data_validation" / "drift_report" / "report.yaml")

    # Fallback: search common paths (limited depth; keeps it lightweight).
    candidate_paths.extend(Path("Artifacts").glob("**/data_validation/drift_report/report.yaml"))

    for p in candidate_paths:
        if p.exists():
            content = yaml.safe_load(p.read_text(encoding="utf-8"))
            rows = []
            for feature, payload in (content or {}).items():
                rows.append(
                    {
                        "feature": feature,
                        "p_value": float(payload.get("p_value", np.nan)),
                        "drift_detected": bool(payload.get("drift_status", False)),
                    }
                )
            return pd.DataFrame(rows).sort_values("p_value", ascending=True), p
    return None, None


def _load_latest_test_array() -> Tuple[Optional[np.ndarray], Optional[Path]]:
    latest = _find_latest_artifact_dir()
    candidate_paths = []
    if latest is not None:
        candidate_paths.append(latest / "data_transformation" / "transformed" / "test.npy")
    candidate_paths.extend(Path("Artifacts").glob("**/data_transformation/transformed/test.npy"))

    for p in candidate_paths:
        if p.exists():
            arr = np.load(p)
            return arr, p
    return None, None


@st.cache_resource
def _load_champion_model():
    """
    Load the current champion model.

    TODO: Replace with a real model registry lookup (MLflow Model Registry, etc.)
    """
    import pickle

    path = Path("final_model") / "model.pkl"
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


@st.cache_data
def _load_schema_feature_names() -> Optional[list[str]]:
    schema_path = Path("data_schema") / "schema.yaml"
    if not schema_path.exists():
        return None
    content = yaml.safe_load(schema_path.read_text(encoding="utf-8")) or {}
    cols = content.get("columns", [])
    # columns entries are like "feature: dtype"
    features = []
    for entry in cols:
        if isinstance(entry, str):
            name = entry.split(":")[0].strip()
            if name and name.lower() != "result":
                features.append(name)
    return features or None


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Network Intrusion Detection - Monitoring", layout="wide")
st.title("Network Intrusion Detection - Monitoring")

perf = get_performance_history(days=30)
perf = perf.sort_values("date")

current_f1 = float(perf["f1_score"].iloc[-1])
week_ago_idx = max(0, len(perf) - 8)
delta_f1 = current_f1 - float(perf["f1_score"].iloc[week_ago_idx])

recent_preds = get_recent_predictions(limit=100)
preds_today = int((recent_preds["timestamp"].dt.date == dt.date.today()).sum()) if not recent_preds.empty else 0

# TODO: Replace latency stub with real service telemetry (p95 inference latency).
latency_p95_ms = float(np.percentile(np.random.default_rng(1).uniform(25, 80, size=200), 95))

drift_df, drift_path = _load_latest_drift_report()
drift_detected = bool(drift_df is not None and drift_df["drift_detected"].any())
drift_status_text = "Drift Detected" if drift_detected else "No Drift"

col1, col2, col3, col4 = st.columns(4)
col1.metric("F1 Score (current)", f"{current_f1:.3f}", f"{delta_f1:+.3f} vs last week")
col2.metric("Predictions Today", f"{preds_today}")
col3.metric("Latency (p95)", f"{latency_p95_ms:.0f} ms")
col4.metric("Drift Status", drift_status_text)

st.divider()

st.subheader("Performance Over Time")
st.line_chart(perf.set_index("date")[["f1_score", "precision", "recall"]])

st.divider()

st.subheader("Feature Importance (SHAP)")

model = _load_champion_model()
test_arr, test_arr_path = _load_latest_test_array()
feature_names = _load_schema_feature_names()

if model is None:
    st.info("No `final_model/model.pkl` found yet. Train the pipeline once, then refresh this page.")
elif test_arr is None:
    st.info("No transformed `test.npy` found in `Artifacts/`. Run the training pipeline once, then refresh.")
else:
    try:
        import matplotlib.pyplot as plt
        import shap

        X_test = test_arr[:, :-1]
        n = X_test.shape[0]
        sample_size = int(min(500, n))
        rng = np.random.default_rng(123)
        sample_idx = rng.choice(n, size=sample_size, replace=False) if n > sample_size else np.arange(n)
        X_sample = X_test[sample_idx]

        # TreeExplainer assumption (works for many sklearn tree models).
        explainer = shap.TreeExplainer(model)
        shap_exp = explainer(X_sample)

        st.caption(
            f"Using SHAP TreeExplainer on a sample of {len(X_sample)} rows "
            f"from `{test_arr_path}`. TODO: Wire to real holdout data / production samples."
        )

        # (1) Global bar plot of feature importance.
        plt.figure()
        shap.summary_plot(
            shap_exp.values,
            X_sample,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
            max_display=20,
        )
        st.pyplot(plt.gcf(), clear_figure=True)

        # (2) Individual prediction explanation (waterfall).
        st.markdown("**Individual Explanation**")
        idx = st.slider("Select sample index", min_value=0, max_value=len(X_sample) - 1, value=0, step=1)
        plt.figure()
        shap.plots.waterfall(shap_exp[idx], show=False, max_display=15)
        st.pyplot(plt.gcf(), clear_figure=True)

    except Exception as e:
        st.warning(
            "SHAP section failed to render. This can happen if the saved model is not tree-based "
            "or if SHAP isn't installed yet."
        )
        st.code(str(e))
        st.info("TODO: If your model isn't tree-based, switch to `shap.KernelExplainer` (slower) or log SHAP values offline.")

st.divider()

st.subheader("Data Drift Visualization (KS Test)")

if drift_df is None:
    st.info("No drift report found. Expected a file like `Artifacts/.../data_validation/drift_report/report.yaml`.")
else:
    import matplotlib.pyplot as plt

    st.caption(f"Loaded drift report from `{drift_path}`.")
    drift_df = drift_df.copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(drift_df["feature"], drift_df["p_value"])
    ax.axhline(0.05, color="red", linestyle="--", linewidth=2, label="p=0.05 threshold")
    ax.set_title("KS-test p-values by feature")
    ax.set_ylabel("p-value")
    ax.set_xticklabels(drift_df["feature"], rotation=90)
    ax.legend(loc="upper right")
    st.pyplot(fig, clear_figure=True)

    drift_features = drift_df.loc[drift_df["drift_detected"], "feature"].tolist()
    if drift_features:
        st.warning(f"Drift detected in {len(drift_features)} feature(s): {', '.join(drift_features[:25])}")
    else:
        st.success("No drift detected (all features p-value >= 0.05).")

st.divider()

st.subheader("Recent Predictions")
st.dataframe(recent_preds, use_container_width=True, hide_index=True)

