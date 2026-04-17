# SSC Estimator — Lower Mississippi River

An interactive web app for estimating **Suspended Sediment Concentration (SSC)** from Sentinel-2 satellite imagery using a two-stage machine learning pipeline.

**Live Demo**: [huggingface.co/spaces/eaadeniyi/ssc-estimator-mississippi](https://huggingface.co/spaces/eaadeniyi/ssc-estimator-mississippi)

---

## Overview

This app is part of a Master's thesis on satellite-based SSC estimation in the Lower Mississippi River. It implements a two-stage pipeline:

- **Stage 1**: Predicts turbidity (NTU) from Sentinel-2 spectral reflectance using 8 machine learning models
- **Stage 2**: Converts turbidity to SSC (mg/L) via a physically-based power law relationship

```
Sentinel-2 Reflectance (ACOLITE Rrs)
        ↓
Stage 1: Reflectance → Turbidity (NTU)
        ↓
Stage 2: SSC = 1.5259 × Turb^1.1093 × 1.0307
        ↓
SSC (mg/L)
```

---

## Models

Eight models are evaluated using a dual-approach strategy:

| Model | Type | Approach | Test R² |
|-------|------|----------|---------|
| Linear Regression | Linear | Log-transform + Duan's smearing | 0.832 |
| Ridge | Linear | Log-transform + Duan's smearing | **0.835** |
| ElasticNet | Linear | Log-transform + Duan's smearing | 0.808 |
| SVR | Non-linear | Direct prediction (18 features) | 0.647 |
| Random Forest | Non-linear | Direct prediction (18 features) | 0.814 |
| XGBoost | Non-linear | Direct prediction (18 features) | 0.729 |
| ANN | Deep learning | Direct prediction (18 features) | — |
| CNN 1D | Deep learning | Direct prediction (18 features) | 0.683 |

End-to-end SSC performance (independent test, n=15): **SVR R²=0.881**

---

## Input

The app accepts four Sentinel-2 band values processed through **ACOLITE atmospheric correction** (Rrs, sr⁻¹):

| Band | Sentinel-2 Band | Wavelength |
|------|----------------|------------|
| Blue | B2 | ~490 nm |
| Green | B3 | ~560 nm |
| Red | B4 | ~665 nm |
| NIR | B8 | ~842 nm |

All 18 spectral features (band ratios, indices) are computed automatically from these four inputs.

> **Note**: Models were trained on ACOLITE-processed imagery. Using GEE/Sen2Cor reflectances will reduce accuracy.

---

## Usage

### Single Point
Enter the four Rrs band values manually. The app returns turbidity and SSC predictions from all 8 models.

### Batch CSV
Upload a CSV file with columns: `blue`, `green`, `red`, `nir`. The app processes every row and returns a full results table.

Example CSV format:
```
blue,green,red,nir
0.020,0.025,0.030,0.015
0.018,0.022,0.035,0.020
```

---

## Training Data

- **Stage 1**: 70 ACOLITE-processed Sentinel-2 scenes matched with in-situ turbidity measurements at Belle Chasse, LA (USGS gauge 07374525), 2017–2024
- **Stage 2**: 102 turbidity–SSC pairs (cleaned) from the same station
- **Training range**: 9–119 NTU turbidity | 8–400 mg/L SSC

---

## Study Area

**Belle Chasse, Louisiana** — Lower Mississippi River, approximately 20 km south of New Orleans. This station is the reference gauge for the **Mid-Barataria Sediment Diversion**, a $3 billion coastal restoration project.

---

## Citation

If you use this app or the methodology in your work, please cite:

> Adeniyi, E. (2026). *Satellite-Based Suspended Sediment Concentration Estimation in the Lower Mississippi River Using Sentinel-2 Imagery and a Two-Stage Machine Learning Pipeline*. Master's Thesis, Louisiana State University.

---

## References

- Vanhellemont & Ruddick (2018): ACOLITE atmospheric correction for water applications
- Duan (1983): Smearing estimate — nonparametric retransformation, *JASA* 78(383), 605–610
- Stull & Ahmari (2024): SSC estimation, Lower Brazos River, SVM/ANN/ELM
