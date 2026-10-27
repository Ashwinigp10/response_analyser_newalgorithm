# MindWave ML Evaluation Report: XGBoost Screening Models

## 1. Executive Summary
This report documents the end-to-end development, training, and held-out test evaluation of the **XGBoost classification models** powering the **MindWave Response Analyzer**. The models are strictly based on **XGBoost (XGBClassifier)**, replacing all previous classification approaches.

The models address two screening tasks:
1. **PQ-16 Psychosis Prodrome Screening**: Classifies whether an individual's response pattern crosses the established clinical screening threshold (score >= 6 on the 16-item Prodromal Questionnaire).
2. **MDQ Bipolar/Hypomania Screening**: Classifies whether an individual meets the Hirschfeld et al. clinical screening criteria (7+ co-occurring symptoms with moderate/serious functional impact).

---

## 2. Dataset Attribution, Source & Provenance
* **Dataset**: Prevalence of Psychiatric Symptoms in UK University Students (Akram et al., 2023)
* **Official Publication**: Akram, U., Irvine, K., Gardani, M., Allen, S., & Stevenson, J. C. (2023). *Scientific Data* (Nature Portfolio).
* **Official DOI**: [https://doi.org/10.6084/m9.figshare.24052236.v2](https://doi.org/10.6084/m9.figshare.24052236.v2)
* **Licence**: Creative Commons Attribution 4.0 International (CC BY 4.0)
* **Participants**: N = 1,408 UK university students
* **Integrity**: Acquired directly via Figshare API; SHA256 checksums verified; zero synthetic or fabricated records.

---

## 3. Data Schema and Preprocessing
### PQ-16 Features:
* 16 binary question response indicators: `PRO16_1` through `PRO16_16` (0 = False, 1 = True).
* Minimal missing data (<0.7%) imputed with 0 (unendorsed).
* Target label: Binary indicator of Ultra-High Risk screening threshold (`P16_Psychotic_Exp_Sum >= 6`).

### MDQ Features:
* 13 symptom indicators: `MDQ1_1` through `MDQ1_13` (0 = No, 1 = Yes).
* 1 synchrony indicator: `MDQ_Frequency` (did symptoms co-occur, 0 = No, 1 = Yes).
* 1 functional severity indicator: `MDQ3_Severity` (1 = No problem, 2 = Minor, 3 = Moderate, 4 = Serious).
* Target label: Hirschfeld et al. (2000) 3-part clinical screening rule (`MDQ1 sum >= 7` AND `MDQ_Frequency == 1` AND `MDQ3_Severity >= 3`).

### Splitting Strategy:
* **Train**: 70% (N = 985)
* **Validation**: 15% (N = 211) — used solely for early stopping
* **Held-out Test**: 15% (N = 212) — completely isolated from model fitting and hyperparameter selection
* **Stratification**: Preserved positive class ratio across all splits. Random seed: `42`.

---

## 4. XGBoost Model Configuration
* **Algorithm**: `xgboost.XGBClassifier` (v3.4.1)
* **PQ-16 Hyperparameters**:
  * `max_depth`: 3
  * `learning_rate`: 0.05
  * `n_estimators`: 200 (Best iteration: 199)
  * `subsample`: 0.85
  * `colsample_bytree`: 0.85
  * `reg_alpha`: 0.1, `reg_lambda`: 1.0
  * `scale_pos_weight`: 2.38 (class imbalance compensation)
* **MDQ Hyperparameters**:
  * `max_depth`: 3
  * `learning_rate`: 0.05
  * `n_estimators`: 200 (Best iteration: 192)
  * `subsample`: 0.85
  * `colsample_bytree`: 0.85
  * `reg_alpha`: 0.2, `reg_lambda`: 1.2
  * `scale_pos_weight`: 7.79 (class imbalance compensation)

---

## 5. Quantitative Evaluation Results (Held-Out Test Set, N=212)

### A. PQ-16 Psychosis Prodrome Model
| Metric | 5-Fold CV (Train) | Held-Out Test Set |
|---|---|---|
| **Accuracy** | 0.9584 +/- 0.0149 | **0.9623** |
| **Precision** | 0.9140 +/- 0.0459 | **0.9365** |
| **Recall / Sensitivity** | 0.9521 +/- 0.0363 | **0.9365** |
| **Specificity** | N/A | **0.9732** |
| **F1-Score** | 0.9315 +/- 0.0243 | **0.9365** |
| **ROC-AUC** | 0.9951 +/- 0.0032 | **0.9924** |

**Confusion Matrix (Held-out Test, N=212)**:
* True Negative (TN): 145
* False Positive (FP): 4
* False Negative (FN): 4
* True Positive (TP): 59

---

### B. MDQ Bipolar Screening Model
| Metric | 5-Fold CV (Train) | Held-Out Test Set |
|---|---|---|
| **Accuracy** | 0.9807 +/- 0.0050 | **0.9906** |
| **Precision** | 0.8619 +/- 0.0343 | **0.9231** |
| **Recall / Sensitivity** | 0.9913 +/- 0.0174 | **1.0000** |
| **Specificity** | N/A | **0.9894** |
| **F1-Score** | 0.9216 +/- 0.0191 | **0.9600** |
| **ROC-AUC** | 0.9990 +/- 0.0011 | **1.0000** |

**Confusion Matrix (Held-out Test, N=212)**:
* True Negative (TN): 186
* False Positive (FP): 2
* False Negative (FN): 0
* True Positive (TP): 24

---

## 6. Critical Research Validity & Methodological Disclosures
In strict accordance with clinical research ethics and rigorous machine learning standards:

1. **Circularity / Threshold Reproduction Task**:
   * For the PQ-16 model, the positive screening label is computed directly from whether the participant endorsed >= 6 of the 16 PQ-16 questionnaire items.
   * Consequently, the model is performing **questionnaire-screening threshold reproduction**, learning nonlinear combinations and symptom weighting.
   * **Do not interpret high accuracy or ROC-AUC as an independent clinical diagnosis of schizophrenia**. The dataset does not contain gold-standard psychiatric clinical interview diagnoses (e.g., SCID, CAARMS, or SIPS).
2. **MDQ Screening Criteria vs. Clinical Diagnosis**:
   * The MDQ model learns to identify individuals meeting the multi-criteria Hirschfeld screening threshold (symptom count, episode synchrony, and functional impairment).
   * While 68 participants reported a prior clinical bipolar diagnosis (`MDQ5`), the model is primarily formulated to identify screening risk indicators rather than replacing a comprehensive clinical psychiatric assessment.
3. **Population and Generalisability Limitations**:
   * The source dataset consists of **1,408 UK university students** aged predominantly 18–25.
   * Model behavior **may not generalise** to younger children (6–12), older geriatric adults (50+), or international populations (e.g. Indian or non-Western cohorts) where symptom expression, stigma, and language nuances differ.
   * MindWave provides age-adapted question phrasing; however, the model should be treated as an experimental triage and educational tool.
4. **Clinical Disclaimer**:
   * This tool is an **assistive screening risk analyzer**, NOT a medical diagnostic instrument. It does not replace a clinical interview conducted by a qualified psychiatrist or licensed clinical psychologist.

---

## 7. Model Serialization & Reproducibility
* Models are saved as native JSON (`pq16_xgboost_model.json`, `mdq_xgboost_model.json`) and joblib pipelines.
* Artifacts reside in `ml/models/`.
* Complete reproducibility is guaranteed via pinned random seed (42) and isolated data splits.
