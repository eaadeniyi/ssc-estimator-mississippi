"""
SSC Estimation App — Mississippi River
Two-stage pipeline: Sentinel-2 Reflectance → Turbidity → SSC
Primary model: SVR (best end-to-end SSC R²=0.881 on independent test set)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import gradio as gr
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── Paths ──
STAGE1_DIR  = Path("models/stage1")
STAGE2_JSON = Path("models/stage2/best_stage2_cleaned.json")

# ── Stage 2 parameters ──
s2    = json.loads(STAGE2_JSON.read_text())
S2_A  = s2["a"]
S2_B  = s2["b"]
S2_SF = s2["smearing"]["smearing_factor"]

# ── Load SVR (primary model) ──
SVR_MODEL  = joblib.load(STAGE1_DIR / "svr_model.joblib")
SVR_SCALER = joblib.load(STAGE1_DIR / "svr_scaler.joblib")

FEATURE_ORDER = [
    "blue", "green", "red", "nir", "ndvi", "ndti", "nsmi", "ndssi",
    "red-nir", "sed.index", "nir/red", "nir/blue", "nir/green",
    "red/green", "(r/g)+nir", "red-squared", "nir-squared", "log(nir)/log(red)"
]


def compute_features(blue, green, red, nir):
    eps = 1e-10
    return {
        "blue": blue, "green": green, "red": red, "nir": nir,
        "ndvi":  (nir - red)   / (nir + red   + eps),
        "ndti":  (red - green) / (red + green  + eps),
        "nsmi":  (red + green - blue) / (red + green + blue + eps),
        "ndssi": (blue - nir)  / (blue + nir   + eps),
        "red-nir":   red - nir,
        "sed.index": (red - nir) / (red + nir + eps),
        "nir/red":   nir / (red   + eps),
        "nir/blue":  nir / (blue  + eps),
        "nir/green": nir / (green + eps),
        "red/green": red / (green + eps),
        "(r/g)+nir": (red / (green + eps)) + nir,
        "red-squared": red ** 2,
        "nir-squared": nir ** 2,
        "log(nir)/log(red)": np.log(nir + eps) / (np.log(red + eps) if red > 0 else eps),
    }


def predict_ssc(turb):
    return float((10 ** (S2_A * np.log10(max(turb, 0.1)) + S2_B)) * S2_SF)


def run_svr(blue, green, red, nir):
    feats = compute_features(blue, green, red, nir)
    X = np.array([[feats[f] for f in FEATURE_ORDER]])
    X_sc = SVR_SCALER.transform(X)
    turb = float(max(SVR_MODEL.predict(X_sc)[0], 0.1))
    ssc  = predict_ssc(turb)

    if turb < 9 or turb > 119:
        warn = gr.update(visible=True,
                         value=f"Warning: predicted turbidity ({turb:.1f} NTU) is outside training range (9–119 NTU). Interpret with caution.")
    else:
        warn = gr.update(visible=False, value="")

    return round(turb, 2), round(ssc, 2), warn


def run_svr_csv(file):
    if file is None:
        return "Please upload a CSV file.", None

    df = pd.read_csv(file.name, encoding="utf-8-sig")
    df.columns = df.columns.str.lower()
    missing = {"blue", "green", "red", "nir"} - set(df.columns)
    if missing:
        return f"Missing columns: {missing}. CSV must have: blue, green, red, nir", None

    rows = []
    for i, row in df.iterrows():
        feats = compute_features(row["blue"], row["green"], row["red"], row["nir"])
        X = np.array([[feats[f] for f in FEATURE_ORDER]])
        X_sc = SVR_SCALER.transform(X)
        turb = float(max(SVR_MODEL.predict(X_sc)[0], 0.1))
        ssc  = predict_ssc(turb)
        flag = "OUT OF RANGE" if turb < 9 or turb > 119 else "OK"
        rows.append({"Row": i + 1,
                     "Turbidity (NTU)": round(turb, 2),
                     "SSC (mg/L)":      round(ssc, 2),
                     "Range Check":     flag})

    return "Done.", pd.DataFrame(rows)


# ── UI ──
with gr.Blocks(title="SSC Estimator — Mississippi River") as demo:
    gr.Markdown("""
    # SSC Estimator — Lower Mississippi River
    **Two-stage pipeline**: Sentinel-2 Reflectance (ACOLITE) → Turbidity → SSC (mg/L)

    **Model**: Support Vector Regression (SVR) — best end-to-end SSC performance (R² = 0.881, RMSE = 14.19 mg/L)
    on an independent test set of 15 scenes.

    **Stage 2**: `SSC = 1.5259 × Turbidity^1.1093 × 1.0307`

    > Input values must be **Rrs (sr⁻¹)** from **ACOLITE atmospheric correction** (not GEE/Sen2Cor).
    > Training range: 9–119 NTU | 8–400 mg/L
    """)

    with gr.Tabs():
        # ── Tab 1: Single point ──
        with gr.Tab("Single Point"):
            gr.Markdown("Enter the four ACOLITE Rrs band values for one pixel or location.")
            with gr.Row():
                inp_blue  = gr.Number(label="Blue  (B2, Rrs sr⁻¹)", value=0.020)
                inp_green = gr.Number(label="Green (B3, Rrs sr⁻¹)", value=0.025)
                inp_red   = gr.Number(label="Red   (B4, Rrs sr⁻¹)", value=0.030)
                inp_nir   = gr.Number(label="NIR   (B8, Rrs sr⁻¹)", value=0.015)

            btn = gr.Button("Predict SSC", variant="primary")

            with gr.Row():
                out_turb = gr.Number(label="Predicted Turbidity (NTU)", interactive=False)
                out_ssc  = gr.Number(label="Predicted SSC (mg/L)",      interactive=False)

            out_warn = gr.Textbox(label="Range Warning", interactive=False, visible=False)

            btn.click(run_svr,
                      inputs=[inp_blue, inp_green, inp_red, inp_nir],
                      outputs=[out_turb, out_ssc, out_warn])

        # ── Tab 2: CSV batch ──
        with gr.Tab("Batch CSV Upload"):
            gr.Markdown("""
            Upload a CSV with columns: `blue`, `green`, `red`, `nir` (Rrs sr⁻¹, ACOLITE).
            One row per observation. SVR predictions are returned for all rows.

            **Example CSV:**
            ```
            blue,green,red,nir
            0.020,0.025,0.030,0.015
            0.018,0.022,0.035,0.020
            ```
            """)
            file_input = gr.File(label="Upload CSV", file_types=[".csv"])
            run_btn    = gr.Button("Run Predictions", variant="primary")
            status_msg = gr.Textbox(label="Status", interactive=False)
            out_csv    = gr.Dataframe(label="Results")

            run_btn.click(run_svr_csv,
                          inputs=file_input,
                          outputs=[status_msg, out_csv])

    gr.Markdown("""
    ---
    **Study area**: Belle Chasse, LA — USGS gauge 07374525, Lower Mississippi River
    **Training data**: 70 ACOLITE Sentinel-2 scenes (2017–2024)
    **Thesis**: Adeniyi, E. (2026). Louisiana State University.
    """)

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
