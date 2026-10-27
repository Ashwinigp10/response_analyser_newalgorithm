# MindWave Response Analyzer (XGBoost Only Implementation)

MindWave Response Analyzer is an intelligent mental health response screening platform integrating validated clinical psychiatric questionnaires (**PQ-16** for prodromal psychosis risk and **MDQ** for bipolar spectrum screening) with high-performance **XGBoost (XGBClassifier)** gradient-boosted decision tree models.

---

## 1. Project Architecture

```
MINDWAVE-RESPONSE-ANALYZER-main/
│
├── ml/                                 # Complete XGBoost Machine Learning Pipeline
│   ├── data/
│   │   ├── raw/                        # Downloaded raw research dataset files & metadata
│   │   │   ├── Akram et al 2023 Student MH Data UK SciData.xlsx
│   │   │   ├── Akram et al 2023 Student MH Data UK SciData.sav
│   │   │   ├── Variable codes and abbreviations .pdf
│   │   │   └── dataset_metadata.json   # Verified SHA256 checksums and source metadata
│   │   └── processed/                  # Stratified train, val, and test splits (CSV)
│   ├── models/                         # Serialized XGBoost models & pipelines
│   │   ├── pq16_xgboost_model.json     # Native XGBoost model for PQ-16
│   │   ├── pq16_xgboost_pipeline.json  # Metadata, feature schema, CV & test metrics
│   │   ├── mdq_xgboost_model.json      # Native XGBoost model for MDQ
│   │   └── mdq_xgboost_pipeline.json   # Metadata, feature schema, CV & test metrics
│   ├── src/
│   │   ├── download_dataset.py         # Automated dataset acquisition from Figshare API
│   │   ├── verify_dataset.py           # Integrity, schema, and distribution verification
│   │   ├── preprocess.py               # Missing-data handling & stratified 70/15/15 splits
│   │   ├── train.py                    # XGBoost training with 5-fold CV & early stopping
│   │   ├── evaluate.py                 # Comprehensive evaluation report generator
│   │   └── predict.py                  # Standalone inference class & schema validator
│   ├── tests/
│   │   ├── test_models.py              # Pytest unit tests for model predictions
│   │   ├── test_api.py                 # Pytest integration tests for FastAPI endpoints
│   │   └── verify_live_system.py       # Live system verification script
│   ├── app.py                          # FastAPI backend service (Port 5000)
│   ├── evaluation_report.md            # In-depth scientific evaluation report
│   └── requirements.txt                # Pinned Python dependencies
│
├── src/                                # React 19 Frontend Application
│   ├── context/
│   │   ├── ScreeningContext.jsx        # Manages questionnaire flow & queries /api/ml/predict
│   │   ├── AuthContext.jsx             # Session authentication state
│   │   └── JournalContext.jsx          # Mood journal tracking
│   ├── pages/
│   │   ├── ResultPage.jsx              # Displays questionnaire scores & XGBoost ML card
│   │   ├── QuestionnairePage.jsx       # Adaptive age-specific questionnaire flow
│   │   └── ...                         # Dashboard, Care Locator, Auth, Journal
│   └── data/
│       ├── screeningData.js            # Age groups (non-overlapping) & adaptive questions
│       ├── pq16Questions.js            # Standard PQ-16 item definitions
│       └── bipolarQuestions.js         # Standard MDQ item definitions
│
├── setup.ps1                           # Automated end-to-end PowerShell setup script
├── run_backend.ps1                     # Starts FastAPI backend (Port 5000)
├── run_frontend.ps1                    # Starts React Vite dev server (Port 3000)
├── run_all.ps1                         # Starts both backend and frontend
└── package.json                        # Frontend dependencies and Vite configuration
```

---

## 2. Research Dataset Provenance & Attribution

* **Dataset Title**: *Comprehensive Data on the Prevalence of Psychiatric Symptoms in UK University Students: Data Files SPSS and XLSX Format and Variable Codes*
* **Authors**: Umair Akram, Kamila Irvine, Maria Gardani, Sarah Allen, Jodie C. Stevenson (2023)
* **Published In**: *Scientific Data* (Nature Portfolio)
* **DOI**: [https://doi.org/10.6084/m9.figshare.24052236.v2](https://doi.org/10.6084/m9.figshare.24052236.v2)
* **Licence**: Creative Commons Attribution 4.0 International (**CC BY 4.0**)
* **Cohort**: N = 1,408 UK university students
* **Dataset Acquisition**: Automatically downloaded directly via the Figshare API (`ml/src/download_dataset.py`) with SHA256 integrity verification. Zero synthetic records.

---

## 3. XGBoost Model Performance (Held-Out Test Set, N=212)

Models were evaluated on a completely held-out test split (15% of the dataset) that was never used during model training or hyperparameter tuning.

| Metric | PQ-16 Psychosis Prodrome Screening | MDQ Bipolar Screening |
|---|---|---|
| **Algorithm** | `XGBClassifier` (XGBoost v3.4+) | `XGBClassifier` (XGBoost v3.4+) |
| **5-Fold CV Accuracy (Train)** | 0.9584 ± 0.0149 | 0.9807 ± 0.0050 |
| **Held-Out Test Accuracy** | **0.9623** (96.23%) | **0.9906** (99.06%) |
| **Sensitivity / Recall** | **0.9365** (93.65%) | **1.0000** (100.0%) |
| **Specificity** | **0.9732** (97.32%) | **0.9894** (98.94%) |
| **F1-Score** | **0.9365** | **0.9600** |
| **ROC-AUC** | **0.9924** | **1.0000** |
| **Confusion Matrix (TN/FP/FN/TP)** | TN: 145, FP: 4, FN: 4, TP: 59 | TN: 186, FP: 2, FN: 0, TP: 24 |

---

## 4. Critical Research Validity & Disclosures

1. **Questionnaire Screening Threshold Reproduction**:
   * For PQ-16, the positive label is defined by the standard clinical screening cutoff of $\ge 6$ endorsed symptoms (Ising et al., 2012). Because the target is derived from the questionnaire items themselves, the model is performing **threshold-reproduction classification** rather than an independent clinical diagnosis.
   * For MDQ, the positive label is defined by the standard Hirschfeld et al. (2000) 3-part criterion (7+ symptoms, co-occurrence, and moderate/severe functional impairment).
   * **Do not interpret high accuracy as an independent clinical diagnosis of schizophrenia or bipolar disorder**.
2. **Population Limitations**:
   * The underlying dataset is derived from UK higher education students (young adults aged 18–25).
   * Predictions may not generalize to children (6–12), older adults (51+), or non-Western populations without external calibration.
3. **Medical Disclaimer**:
   * MindWave is an educational and supportive screening analysis tool. It is **not a diagnostic instrument** and cannot replace an in-person clinical evaluation by a qualified psychiatrist or licensed clinical psychologist.

---

## 5. API Endpoints (FastAPI Backend, Port 5000)

| Endpoint | Method | Description |
|---|---|---|
| `/api/ml/health` | `GET` | Reports service status, XGBoost version, and model load states |
| `/api/ml/model-info` | `GET` | Returns model hyperparameters, CV scores, test metrics, and attribution |
| `/api/ml/predict` | `POST` | Executes real-time XGBoost inference given questionnaire responses |
| `/api/auth/register` | `POST` | User registration |
| `/api/auth/login` | `POST` | User login |
| `/api/auth/profile` | `GET` | Current user profile |
| `/api/screenings` | `GET`, `POST` | Retrieve and save screening history |
| `/api/screenings/clear` | `POST` | Clear screening history |
| `/api/moods` | `GET`, `POST`, `DELETE` | Mood journal entries |

---

## 6. Quick Start & Execution Guide (Windows PowerShell)

### Automated Setup
To run the complete pipeline (check versions, create `.venv`, install dependencies, download dataset, preprocess, train XGBoost models, verify artifacts, and run tests):

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

To explicitly force retraining:
```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1 -ForceRetrain
```

### Starting the Applications

**Option 1: Start Both Services Simultaneously**
```powershell
powershell -ExecutionPolicy Bypass -File run_all.ps1
```

**Option 2: Start Separately**
* Terminal 1 (Backend):
  ```powershell
  powershell -ExecutionPolicy Bypass -File run_backend.ps1
  # Alternatively: .\.venv\Scripts\python.exe ml\app.py
  ```
* Terminal 2 (Frontend):
  ```powershell
  powershell -ExecutionPolicy Bypass -File run_frontend.ps1
  # Alternatively: npm run dev
  ```

* Access Backend Health: [http://127.0.0.1:5000/api/ml/health](http://127.0.0.1:5000/api/ml/health)
* Access Frontend UI: [http://localhost:3000](http://localhost:3000)

### Running Automated Tests
```powershell
.\.venv\Scripts\python.exe -m pytest ml\tests -v
.\.venv\Scripts\python.exe ml\tests\verify_live_system.py
```
