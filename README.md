# Pulsar-Net 🔭

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?logo=jupyter)

> High-precision pulsar detection from radio survey data using physics-informed feature engineering and cost-sensitive gradient boosting.

---

## Overview

Pulsars are rapidly rotating neutron stars that are extremely rare in radio survey data, making up less than 10% of candidates. The rest is Radio Frequency Interference (RFI) and noise.

**Pulsar-Net** is an end-to-end machine learning pipeline trained on the [HTRU2 dataset](https://www.kaggle.com/datasets/charitarth/pulsar-dataset-htru2) that distinguishes genuine pulsar signals from terrestrial interference. The pipeline expands the original 8 statistical features to **19 physics-informed features**, then trains a tuned XGBoost classifier with threshold optimization and full SHAP interpretability.

---

## Results

| Metric | Score |
|:---|:---|
| Accuracy | **98%** |
| F1-Score (Pulsar class) | **0.90** |
| Precision (Pulsar) | 0.90 |
| Recall (Pulsar) | 0.90 |
| CV Mean F1 (5-Fold) | **0.877** |
| Balanced Threshold | **0.6495** |
| High-confidence precision (>=0.9) | **93.7%** |

Validated using 5-Fold Stratified Cross-Validation to ensure consistent performance across the class-imbalanced dataset (9.16% pulsar rate).

---

## Feature Engineering: 8 -> 19 Features

The original HTRU2 dataset provides 8 statistical descriptors of the pulse profile and dispersion measure (DM) curve. Three tiers of domain-informed features were engineered on top of these.

### Tier 1 — Core Physics Ratios

**`snr_robustness`**
Measures how well a signal maintains its signal-to-noise ratio across varying levels of dispersion. Pulsars are physically stable emitters, so a high and consistent SNR is a strong discriminator against sporadic RFI.

**`peak_to_noise`**
Compares the peak signal amplitude to the background noise floor. Human-made interference tends to be erratic and wideband; a clean, narrow peak relative to noise is characteristic of a genuine pulsar.

**`sharpness_index`**
Targets the extremely narrow, high-intensity spikes that are the hallmark of a rotating neutron star. Computed from the kurtosis and standard deviation of the pulse profile, it directly encodes the "peakiness" that physically distinguishes pulsars from diffuse noise.

**`ism_factor`**
Models how the signal has interacted with the Interstellar Medium (ISM) during propagation. Pulsars have a characteristic dispersion pattern imposed by free electrons in the ISM; this feature captures that imprint using the mean and skewness of the DM curve.

### Tier 2 — Advanced Pulsar Signatures

**`pulsar_signature_score`**
A composite score built from the kurtosis of the pulse profile and the sharpness index. It amplifies the joint statistical fingerprint that separates a real pulsar from RFI, giving the model a single high-signal feature that encodes both profile shape and intensity.

**`energy_concentration`**
Quantifies how tightly the signal energy is focused within the pulse window. A pulsar's energy arrives in a precise, narrow burst tied to its rotation period. This feature measures that concentration directly from the mean profile and sharpness index.

**`log_dm_skew`**
A log-normalized measure of how asymmetric the dispersion measure distribution is. The DM skew captures subtle differences in how the pulse has spread through the ISM, and the log transform stabilizes the extreme range of values seen in real survey data.

### Tier 3 — Log-Normalizing Transforms

Radio survey distributions are heavily right-skewed, which can compress important variation at the low end of the scale. Logarithmic transforms were applied to three features to recover this resolution and help the model distinguish subtle signal differences:

`log_kurtosis_profile`, `log_sharpness_index`, `log_pulsar_signature_score`

---

## Model Architecture

- **Algorithm:** XGBoost (`XGBClassifier`)
- **Class imbalance handling:** `scale_pos_weight=10` (cost-sensitive learning)
- **Scaling:** `RobustScaler` (robust to outliers in radio data)
- **Validation:** Stratified 5-Fold Cross-Validation
- **Threshold tuning:** Grid search over [0.5, 0.9] to minimize |FP - FN|

---

## Interpretability (SHAP)

SHAP analysis confirms the model's top features align with known astrophysical theory:

1. `kurtosis_profile` — pulse "peakiness" is the strongest discriminator
2. `log_kurtosis_profile`
3. `pulsar_signature_score`
4. `sharpness_index`
5. `energy_concentration`

This confirms the model is learning genuine physical signal characteristics rather than overfitting to survey-specific artifacts.

---

## Repository Structure

```bash
pulsar-net/
├── data/
│   └── HTRU_2.csv
├── models/
│   ├── pulsar_xgb_model.pkl
│   └── robust_scaler.pkl
├── pulsar_net.ipynb
├── README.md
└── requirements.txt
```

---

## Quickstart

```bash
git clone https://github.com/your-username/pulsar-net.git
cd pulsar-net
pip install -r requirements.txt
jupyter notebook pulsar_net.ipynb
```

---

## Dataset

The [HTRU2 dataset](https://www.kaggle.com/datasets/charitarth/pulsar-dataset-htru2) was collected during the High Time Resolution Universe Survey. It contains 17,898 candidates, of which 1,639 are confirmed pulsars.

> R. J. Lyon et al., "Fifty Years of Pulsar Candidate Selection", MNRAS, 2016.

---

## License

MIT