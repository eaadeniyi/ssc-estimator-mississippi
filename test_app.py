"""Quick smoke test — runs without starting the Gradio server."""
import warnings, os
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json, joblib, numpy as np
from pathlib import Path

STAGE1_DIR = Path("models/stage1")
s2 = json.loads(Path("models/stage2/best_stage2_cleaned.json").read_text())
S2_A, S2_B = s2["a"], s2["b"]
S2_SF = s2["smearing"]["smearing_factor"]

def compute_features(blue, green, red, nir):
    eps = 1e-10
    return {
        "blue": blue, "green": green, "red": red, "nir": nir,
        "ndvi":  (nir - red)  / (nir + red  + eps),
        "ndti":  (red - green) / (red + green + eps),
        "nsmi":  (red + green - blue) / (red + green + blue + eps),
        "ndssi": (blue - nir) / (blue + nir + eps),
        "red-nir": red - nir,
        "sed.index": (red - nir) / (red + nir + eps),
        "nir/red":   nir / (red + eps),
        "nir/blue":  nir / (blue + eps),
        "nir/green": nir / (green + eps),
        "red/green": red / (green + eps),
        "(r/g)+nir": (red / (green + eps)) + nir,
        "red-squared": red ** 2,
        "nir-squared": nir ** 2,
        "log(nir)/log(red)": np.log(nir + eps) / (np.log(red + eps) if red > 0 else eps),
    }

FEATURE_ORDER = [
    "blue","green","red","nir","ndvi","ndti","nsmi","ndssi",
    "red-nir","sed.index","nir/red","nir/blue","nir/green",
    "red/green","(r/g)+nir","red-squared","nir-squared","log(nir)/log(red)"
]

def predict_ssc(turb):
    return float((10 ** (S2_A * np.log10(max(turb, 0.1)) + S2_B)) * S2_SF)

# Test with typical Mississippi River values
blue, green, red, nir = 0.020, 0.025, 0.030, 0.015
feats = compute_features(blue, green, red, nir)
X_all = np.array([[feats[f] for f in FEATURE_ORDER]])

LOG_MODELS = {"linear_regression", "ridge", "elasticnet"}
DISPLAY = {
    "linear_regression":"Linear Regression","ridge":"Ridge","elasticnet":"ElasticNet",
    "svr":"SVR","random_forest":"Random Forest","xgboost":"XGBoost","ann":"ANN","cnn_1d":"CNN 1D"
}

print(f"\n{'Model':<20} {'Turbidity (NTU)':>16} {'SSC (mg/L)':>12}")
print("-" * 52)

for mn in ["linear_regression","ridge","elasticnet","svr","random_forest","xgboost"]:
    try:
        model  = joblib.load(STAGE1_DIR / f"{mn}_model.joblib")
        scaler = joblib.load(STAGE1_DIR / f"{mn}_scaler.joblib")
        if mn in LOG_MODELS:
            params  = json.loads((STAGE1_DIR / f"{mn}_params.json").read_text())
            smear   = json.loads((STAGE1_DIR / f"{mn}_smearing.json").read_text())
            feat_col = params["feature_columns"]
            X_feat  = np.array([[feats[f] for f in feat_col]])
            X_sc    = scaler.transform(X_feat)
            turb    = float(max((10 ** model.predict(X_sc)[0]) * smear["smearing_factor"], 0.1))
        else:
            X_sc = scaler.transform(X_all)
            turb = float(max(model.predict(X_sc)[0], 0.1))
        ssc = predict_ssc(turb)
        print(f"{DISPLAY[mn]:<20} {turb:>16.2f} {ssc:>12.2f}")
    except Exception as e:
        print(f"{DISPLAY[mn]:<20} ERROR: {e}")

# Test DL models
try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    for mn in ["ann", "cnn_1d"]:
        model  = tf.keras.models.load_model(STAGE1_DIR / f"{mn}_model.keras")
        scaler = joblib.load(STAGE1_DIR / f"{mn}_scaler.joblib")
        X_sc   = scaler.transform(X_all)
        if mn == "cnn_1d":
            X_sc = X_sc.reshape(1, 18, 1)
        turb = float(max(model.predict(X_sc, verbose=0).flatten()[0], 0.1))
        ssc  = predict_ssc(turb)
        print(f"{DISPLAY[mn]:<20} {turb:>16.2f} {ssc:>12.2f}")
except Exception as e:
    print(f"DL models skipped: {e}")

print("\nTest complete.")
