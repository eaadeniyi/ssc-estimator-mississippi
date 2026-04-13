"""
SSC Estimation App — Mississippi River
Two-stage pipeline: Sentinel-2 Reflectance → Turbidity → SSC
Models trained on ACOLITE-processed imagery (Belle Chasse, LA)
"""

import json
import warnings
import numpy as np
import pandas as pd
import joblib
import gradio as gr
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Try loading TensorFlow (optional — DL models only) ──
try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    HAS_TF = True
except ImportError:
    HAS_TF = False

# ── Paths ──
STAGE1_DIR = Path("models/stage1")
STAGE2_JSON = Path("models/stage2/best_stage2_cleaned.json")

# ── Stage 2 parameters ──
s2 = json.loads(STAGE2_JSON.read_text())
S2_A  = s2["a"]
S2_B  = s2["b"]
S2_SF = s2["smearing"]["smearing_factor"]

# ── Model definitions ──
LOG_MODELS = {"linear_regression", "ridge", "elasticnet"}
ALL_MODELS = ["linear_regression", "ridge", "elasticnet",
              "svr", "random_forest", "xgboost", "ann", "cnn_1d"]
DISPLAY = {
    "linear_regression": "Linear Regression",
    "ridge":             "Ridge",
    "elasticnet":        "ElasticNet",
    "svr":               "SVR",
    "random_forest":     "Random Forest",
    "xgboost":           "XGBoost",
    "ann":               "ANN",
    "cnn_1d":            "CNN 1D",
}

# ── Load all models at startup ──
MODELS = {}
for mn in ALL_MODELS:
    try:
        if mn in ("ann", "cnn_1d"):
            if not HAS_TF:
                continue
            MODELS[mn] = {
                "model":  tf.keras.models.load_model(STAGE1_DIR / f"{mn}_model.keras"),
                "scaler": joblib.load(STAGE1_DIR / f"{mn}_scaler.joblib"),
                "params": json.loads((STAGE1_DIR / f"{mn}_params.json").read_text()),
            }
        elif mn in LOG_MODELS:
            MODELS[mn] = {
                "model":    joblib.load(STAGE1_DIR / f"{mn}_model.joblib"),
                "scaler":   joblib.load(STAGE1_DIR / f"{mn}_scaler.joblib"),
                "params":   json.loads((STAGE1_DIR / f"{mn}_params.json").read_text()),
                "smearing": json.loads((STAGE1_DIR / f"{mn}_smearing.json").read_text()),
            }
        else:
            MODELS[mn] = {
                "model":  joblib.load(STAGE1_DIR / f"{mn}_model.joblib"),
                "scaler": joblib.load(STAGE1_DIR / f"{mn}_scaler.joblib"),
                "params": json.loads((STAGE1_DIR / f"{mn}_params.json").read_text()),
            }
    except Exception as e:
        print(f"Could not load {mn}: {e}")


# ── Feature engineering ──
def compute_features(blue, green, red, nir):
    eps = 1e-10
    ndvi  = (nir - red)  / (nir + red  + eps)
    ndti  = (red - green) / (red + green + eps)
    nsmi  = (red + green - blue) / (red + green + blue + eps)
    ndssi = (blue - nir) / (blue + nir + eps)
    rg_nir = (red / (green + eps)) + nir
    sed_idx = (red - nir) / (red + nir + eps)
    log_nir_red = np.log(nir + eps) / (np.log(red + eps) if red > 0 else eps)

    return {
        "blue": blue, "green": green, "red": red, "nir": nir,
        "ndvi": ndvi, "ndti": ndti, "nsmi": nsmi, "ndssi": ndssi,
        "red-nir": red - nir,
        "sed.index": sed_idx,
        "nir/red": nir / (red + eps),
        "nir/blue": nir / (blue + eps),
        "nir/green": nir / (green + eps),
        "red/green": red / (green + eps),
        "(r/g)+nir": rg_nir,
        "red-squared": red ** 2,
        "nir-squared": nir ** 2,
        "log(nir)/log(red)": log_nir_red,
    }

FEATURE_ORDER = [
    "blue", "green", "red", "nir", "ndvi", "ndti", "nsmi", "ndssi",
    "red-nir", "sed.index", "nir/red", "nir/blue", "nir/green",
    "red/green", "(r/g)+nir", "red-squared", "nir-squared", "log(nir)/log(red)"
]


def predict_ssc(turb):
    return float((10 ** (S2_A * np.log10(max(turb, 0.1)) + S2_B)) * S2_SF)


def run_all_models(feature_dict):
    """Run all loaded models on a single feature dict. Returns list of result rows."""
    X_all = np.array([[feature_dict[f] for f in FEATURE_ORDER]])
    results = []

    for mn in ALL_MODELS:
        if mn not in MODELS:
            results.append([DISPLAY[mn], "N/A (model not loaded)", "N/A"])
            continue
        try:
            m = MODELS[mn]
            if mn in LOG_MODELS:
                feat_col = m["params"]["feature_columns"]
                X_feat = np.array([[feature_dict[f] for f in feat_col]])
                X_sc = m["scaler"].transform(X_feat)
                log_pred = m["model"].predict(X_sc)[0]
                sf = m["smearing"]["smearing_factor"]
                turb = float(max((10 ** log_pred) * sf, 0.1))
            elif mn == "ann":
                X_sc = m["scaler"].transform(X_all)
                turb = float(max(m["model"].predict(X_sc, verbose=0).flatten()[0], 0.1))
            elif mn == "cnn_1d":
                X_sc = m["scaler"].transform(X_all).reshape(1, 18, 1)
                turb = float(max(m["model"].predict(X_sc, verbose=0).flatten()[0], 0.1))
            else:
                X_sc = m["scaler"].transform(X_all)
                turb = float(max(m["model"].predict(X_sc)[0], 0.1))

            ssc = predict_ssc(turb)
            results.append([DISPLAY[mn], f"{turb:.2f}", f"{ssc:.2f}"])
        except Exception as e:
            results.append([DISPLAY[mn], f"Error: {e}", "N/A"])

    return results


# ── Tab 1: Manual input ──
def predict_manual(blue, green, red, nir):
    feats = compute_features(blue, green, red, nir)
    rows  = run_all_models(feats)
    df = pd.DataFrame(rows, columns=["Model", "Turbidity (NTU)", "SSC (mg/L)"])
    return df


# ── Tab 2: CSV upload ──
def predict_csv(file):
    if file is None:
        return "Please upload a CSV file.", None

    df = pd.read_csv(file.name, encoding="utf-8-sig")
    required = {"blue", "green", "red", "nir"}
    missing = required - set(df.columns.str.lower())
    if missing:
        return f"Missing columns: {missing}. CSV must have: blue, green, red, nir", None

    df.columns = df.columns.str.lower()
    all_rows = []

    for i, row in df.iterrows():
        feats = compute_features(row["blue"], row["green"], row["red"], row["nir"])
        for mn in ALL_MODELS:
            if mn not in MODELS:
                continue
            m = MODELS[mn]
            try:
                X_all = np.array([[feats[f] for f in FEATURE_ORDER]])
                if mn in LOG_MODELS:
                    feat_col = m["params"]["feature_columns"]
                    X_feat = np.array([[feats[f] for f in feat_col]])
                    X_sc = m["scaler"].transform(X_feat)
                    log_pred = m["model"].predict(X_sc)[0]
                    sf = m["smearing"]["smearing_factor"]
                    turb = float(max((10 ** log_pred) * sf, 0.1))
                elif mn == "ann":
                    X_sc = m["scaler"].transform(X_all)
                    turb = float(max(m["model"].predict(X_sc, verbose=0).flatten()[0], 0.1))
                elif mn == "cnn_1d":
                    X_sc = m["scaler"].transform(X_all).reshape(1, 18, 1)
                    turb = float(max(m["model"].predict(X_sc, verbose=0).flatten()[0], 0.1))
                else:
                    X_sc = m["scaler"].transform(X_all)
                    turb = float(max(m["model"].predict(X_sc)[0], 0.1))

                ssc = predict_ssc(turb)
                all_rows.append({"Row": i + 1, "Model": DISPLAY[mn],
                                 "Turbidity (NTU)": round(turb, 2),
                                 "SSC (mg/L)": round(ssc, 2)})
            except Exception as e:
                all_rows.append({"Row": i + 1, "Model": DISPLAY[mn],
                                 "Turbidity (NTU)": None, "SSC (mg/L)": None})

    out_df = pd.DataFrame(all_rows)
    return "Done.", out_df


# ── Build Gradio UI ──
with gr.Blocks(title="SSC Estimator — Mississippi River") as demo:
    gr.Markdown("""
    # SSC Estimator — Lower Mississippi River
    **Two-stage pipeline**: Sentinel-2 Reflectance (ACOLITE) → Turbidity → SSC

    - **Stage 1**: 8 models predict turbidity (NTU) from spectral features
    - **Stage 2**: Power law converts turbidity → SSC (mg/L):
      `SSC = 1.5259 × Turb^1.1093 × 1.0307`
    - Input values must be **Rrs (sr⁻¹)** from **ACOLITE atmospheric correction**
    - Training range: 9–119 NTU | 8–400 mg/L
    """)

    with gr.Tabs():
        # ── Tab 1: Manual ──
        with gr.Tab("Single Point"):
            gr.Markdown("Enter the four ACOLITE Rrs band values for one pixel/location.")
            with gr.Row():
                inp_blue  = gr.Number(label="Blue (B2, Rrs sr⁻¹)",  value=0.020)
                inp_green = gr.Number(label="Green (B3, Rrs sr⁻¹)", value=0.025)
                inp_red   = gr.Number(label="Red (B4, Rrs sr⁻¹)",   value=0.030)
                inp_nir   = gr.Number(label="NIR (B8, Rrs sr⁻¹)",   value=0.015)

            btn = gr.Button("Run All Models", variant="primary")
            out_table = gr.Dataframe(
                headers=["Model", "Turbidity (NTU)", "SSC (mg/L)"],
                label="Predictions — All 8 Models"
            )
            btn.click(predict_manual,
                      inputs=[inp_blue, inp_green, inp_red, inp_nir],
                      outputs=out_table)

        # ── Tab 2: CSV ──
        with gr.Tab("Batch CSV Upload"):
            gr.Markdown("""
            Upload a CSV with columns: `blue`, `green`, `red`, `nir` (Rrs sr⁻¹, ACOLITE).
            Each row is one observation. All 8 models are run for every row.
            """)
            file_input = gr.File(label="Upload CSV", file_types=[".csv"])
            run_btn    = gr.Button("Run Predictions", variant="primary")
            status_msg = gr.Textbox(label="Status", interactive=False)
            out_csv    = gr.Dataframe(label="Results")

            run_btn.click(predict_csv,
                          inputs=file_input,
                          outputs=[status_msg, out_csv])

    gr.Markdown("""
    ---
    **Model training**: 70 ACOLITE Sentinel-2 scenes, Belle Chasse USGS gauge (2017–2024)
    **Best holdout performance**: Ridge R²=0.835 (turbidity), SVR R²=0.881 end-to-end SSC
    """)

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
