import os
import sys
import json
import sqlite3
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import xgboost as xgb

# Force utf-8 for Windows console
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Configure safe logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("mindwave-backend")

SRC_DIR = Path(__file__).resolve().parent
ML_DIR = SRC_DIR.parent if SRC_DIR.name == "src" else SRC_DIR
MODELS_DIR = ML_DIR / "models"
DB_PATH = ML_DIR / "mindwave.db"

# Import predictor
sys.path.insert(0, str(ML_DIR / "src"))
from predict import MindWavePredictor

predictor: Optional[MindWavePredictor] = None


# --- DATABASE SETUP ---
def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cursor = conn.cursor()
    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE,
            password_hash TEXT,
            name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Screenings table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS screenings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            assessment_type TEXT,
            symptom_count INTEGER,
            distress_score INTEGER,
            result_category TEXT,
            age_group TEXT,
            profession TEXT,
            answers TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Moods table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS moods (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            mood TEXT,
            note TEXT,
            date TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


init_db()


# --- LIFESPAN EVENT ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    logger.info("Initializing MindWave ML service...")
    try:
        predictor = MindWavePredictor(MODELS_DIR)
        logger.info("XGBoost models loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load XGBoost models on startup: {e}")
        predictor = None
    yield
    logger.info("Shutting down MindWave ML service...")


app = FastAPI(
    title="MindWave XGBoost ML Backend",
    version="1.0.0",
    description="XGBoost screening inference and application backend for MindWave Response Analyzer",
    lifespan=lifespan
)

# CORS setup for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- PYDANTIC SCHEMAS ---
class PredictRequest(BaseModel):
    assessment_type: str = Field(..., description="'schizophrenia' or 'bipolar'")
    answers: Optional[List[Dict[str, Any]]] = Field(None, description="Array of question answer objects from screening test")
    features: Optional[Dict[str, Any]] = Field(None, description="Direct feature mapping dictionary")
    age_group: Optional[str] = "19-25"
    profession: Optional[str] = "student"


class PredictResponse(BaseModel):
    assessment_type: str
    model_name: str
    algorithm: str
    screening_positive: bool
    probability: float
    risk_level: str
    symptom_count: int
    max_score: int
    threshold_criteria: str
    disclaimer: str


class AuthRegisterRequest(BaseModel):
    name: str
    email: str
    password: str


class AuthLoginRequest(BaseModel):
    email: str
    password: str


class ScreeningSaveRequest(BaseModel):
    assessment_type: str
    symptom_count: int
    distress_score: int
    result_category: str
    age_group: Optional[str] = "19-25"
    profession: Optional[str] = "student"
    answers: Optional[List[Dict[str, Any]]] = None


class MoodSaveRequest(BaseModel):
    mood: str
    note: Optional[str] = ""
    date: Optional[str] = None


# --- AUTH HELPER ---
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def get_current_user_id(authorization: Optional[str] = Header(None)) -> Optional[str]:
    if not authorization:
        return None
    token = authorization.replace("Bearer ", "").strip()
    if not token or token == "null" or token == "undefined":
        return None
    # For demo session, token is user_id or email
    return token


# --- ML API ENDPOINTS ---
@app.get("/api/ml/health")
def ml_health():
    """
    Health check endpoint reporting service status and model availability.
    """
    is_ready = predictor is not None and predictor.is_loaded
    return {
        "status": "ok" if is_ready else "degraded",
        "service": "MindWave XGBoost ML Backend",
        "models_loaded": is_ready,
        "xgboost_version": xgb.__version__,
        "models": {
            "pq16_screening": is_ready and predictor.pq16_model is not None,
            "mdq_screening": is_ready and predictor.mdq_model is not None
        }
    }


@app.get("/api/ml/model-info")
def ml_model_info():
    """
    Returns complete metadata, hyperparameters, and held-out test metrics.
    """
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="XGBoost models are not currently loaded."
        )

    return {
        "framework": "XGBoost",
        "xgboost_version": xgb.__version__,
        "dataset_attribution": {
            "title": "Comprehensive Data on the Prevalence of Psychiatric Symptoms in UK University Students",
            "authors": "Akram, Umair; Irvine, Kamila; Gardani, Maria; Allen, Sarah; Stevenson, Jodie C. (2023)",
            "journal": "Scientific Data (Nature Portfolio)",
            "doi": "https://doi.org/10.6084/m9.figshare.24052236.v2",
            "license": "CC BY 4.0"
        },
        "models": {
            "pq16": predictor.pq16_meta,
            "mdq": predictor.mdq_meta
        },
        "critical_disclosure": (
            "Models perform questionnaire screening threshold reproduction, "
            "not independent biological diagnosis of schizophrenia or bipolar disorder."
        )
    }


@app.post("/api/ml/predict", response_model=PredictResponse)
def ml_predict(req: PredictRequest):
    """
    Performs real-time XGBoost inference for PQ-16 or MDQ screening questionnaire data.
    """
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="XGBoost model service unavailable. Run 'python ml/src/train.py' to generate artifacts."
        )

    test_type = req.assessment_type.lower().strip()
    if test_type not in ["schizophrenia", "pq16", "bipolar", "mdq"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid assessment_type '{req.assessment_type}'. Must be 'schizophrenia' or 'bipolar'."
        )

    # Parse features
    feature_dict = {}
    if req.features:
        feature_dict = dict(req.features)
    elif req.answers:
        # Extract from structured answer list
        for i, ans in enumerate(req.answers):
            q_num = ans.get("id", i + 1)
            is_true = ans.get("answer") == "True" or ans.get("isTrue") is True
            feature_dict[f"PRO16_{q_num}"] = 1 if is_true else 0
            feature_dict[f"MDQ1_{q_num}"] = 1 if is_true else 0
            if q_num == 14:
                feature_dict["MDQ_Frequency"] = 1 if is_true else 0
            if q_num == 11:
                # Severity follow-up or answer
                feature_dict["MDQ3_Severity"] = 3 if is_true else 1

    try:
        if test_type in ["schizophrenia", "pq16"]:
            res = predictor.predict_pq16(feature_dict)
            res["max_score"] = 16
        else:
            res = predictor.predict_mdq(feature_dict)
            res["max_score"] = 15

        return PredictResponse(**res)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


# --- APPLICATION AUTH & PERSISTENCE ENDPOINTS ---
@app.post("/api/auth/register")
def auth_register(req: AuthRegisterRequest):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE email = ?", (req.email.lower().strip(),))
    if cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="User with this email already exists.")

    user_id = "usr_" + hashlib.md5(req.email.encode("utf-8")).hexdigest()[:12]
    pw_hash = hash_password(req.password)
    cursor.execute(
        "INSERT INTO users (id, email, password_hash, name) VALUES (?, ?, ?, ?)",
        (user_id, req.email.lower().strip(), pw_hash, req.name.strip())
    )
    conn.commit()
    conn.close()

    user_data = {"id": user_id, "name": req.name.strip(), "email": req.email.lower().strip()}
    return {"token": user_id, "user": user_data}


@app.post("/api/auth/login")
def auth_login(req: AuthLoginRequest):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email, password_hash FROM users WHERE email = ?", (req.email.lower().strip(),))
    row = cursor.fetchone()
    conn.close()

    if not row or row["password_hash"] != hash_password(req.password):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    user_data = {"id": row["id"], "name": row["name"], "email": row["email"]}
    return {"token": row["id"], "user": user_data}


@app.get("/api/auth/profile")
def auth_profile(token: Optional[str] = Depends(get_current_user_id)):
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email FROM users WHERE id = ?", (token,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return {"id": token, "name": "Guest User", "email": "user@example.com"}
    return {"id": row["id"], "name": row["name"], "email": row["email"]}


@app.get("/api/screenings")
def get_screenings(token: Optional[str] = Depends(get_current_user_id)):
    conn = get_db()
    cursor = conn.cursor()
    if token:
        cursor.execute("SELECT * FROM screenings WHERE user_id = ? ORDER BY id DESC", (token,))
    else:
        cursor.execute("SELECT * FROM screenings ORDER BY id DESC LIMIT 20")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.post("/api/screenings")
def save_screening(req: ScreeningSaveRequest, token: Optional[str] = Depends(get_current_user_id)):
    conn = get_db()
    cursor = conn.cursor()
    uid = token or "anonymous"
    answers_json = json.dumps(req.answers) if req.answers else "[]"
    cursor.execute(
        """
        INSERT INTO screenings (user_id, assessment_type, symptom_count, distress_score, result_category, age_group, profession, answers)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (uid, req.assessment_type, req.symptom_count, req.distress_score, req.result_category, req.age_group, req.profession, answers_json)
    )
    conn.commit()
    conn.close()
    return {"status": "saved"}


@app.post("/api/screenings/clear")
def clear_screenings(token: Optional[str] = Depends(get_current_user_id)):
    conn = get_db()
    cursor = conn.cursor()
    if token:
        cursor.execute("DELETE FROM screenings WHERE user_id = ?", (token,))
    else:
        cursor.execute("DELETE FROM screenings")
    conn.commit()
    conn.close()
    return {"status": "cleared"}


@app.get("/api/moods")
def get_moods(token: Optional[str] = Depends(get_current_user_id)):
    conn = get_db()
    cursor = conn.cursor()
    if token:
        cursor.execute("SELECT * FROM moods WHERE user_id = ? ORDER BY id DESC", (token,))
    else:
        cursor.execute("SELECT * FROM moods ORDER BY id DESC LIMIT 50")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.post("/api/moods")
def save_mood(req: MoodSaveRequest, token: Optional[str] = Depends(get_current_user_id)):
    conn = get_db()
    cursor = conn.cursor()
    uid = token or "anonymous"
    cursor.execute(
        "INSERT INTO moods (user_id, mood, note, date) VALUES (?, ?, ?, ?)",
        (uid, req.mood, req.note, req.date)
    )
    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    return {"id": inserted_id, "status": "saved"}


@app.delete("/api/moods/{mood_id}")
def delete_mood(mood_id: int, token: Optional[str] = Depends(get_current_user_id)):
    conn = get_db()
    cursor = conn.cursor()
    if token:
        cursor.execute("DELETE FROM moods WHERE id = ? AND user_id = ?", (mood_id, token))
    else:
        cursor.execute("DELETE FROM moods WHERE id = ?", (mood_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=5000, reload=False)
