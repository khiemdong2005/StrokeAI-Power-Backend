from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import traceback

# ====== CHATBOT IMPORT ======
import os
from dotenv import load_dotenv
import google.generativeai as genai
# ============================

app = FastAPI(title="Stroke Risk Predictor", version="1.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== PREDICT PART (UNCHANGED) =====================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "logistic_regression_pipeline.joblib"

try:
    pipeline = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Model load failed at {MODEL_PATH}: {e}")

class PredictRequest(BaseModel):
    age: float = Field(..., ge=0, le=120)
    avg_glucose_level: float = Field(..., ge=0)
    bmi: float = Field(..., ge=0)

    gender: str
    hypertension: int = Field(..., ge=0, le=1)
    heart_disease: int = Field(..., ge=0, le=1)
    ever_married: str
    work_type: str
    Residence_type: str
    smoking_status: str

def safe_log1p(x: float) -> float:
    return float(np.log1p(max(float(x), 0.0)))

def to01_str(x: int) -> str:
    return "1" if int(x) == 1 else "0"

def norm_text(s: str) -> str:
    return str(s).strip().lower()

def build_dataframe(p: PredictRequest) -> pd.DataFrame:
    df = pd.DataFrame([{
        "age": float(p.age),
        "avg_glucose_level": float(p.avg_glucose_level),
        "bmi": float(p.bmi),
        "log_avg_glucose_level": safe_log1p(p.avg_glucose_level),
        "log_bmi": safe_log1p(p.bmi),

        "gender": norm_text(p.gender),
        "ever_married": norm_text(p.ever_married),
        "work_type": norm_text(p.work_type),
        "Residence_type": norm_text(p.Residence_type),
        "smoking_status": norm_text(p.smoking_status),

        "hypertension": to01_str(p.hypertension),
        "heart_disease": to01_str(p.heart_disease),
    }])

    cat_cols = [
        "gender", "ever_married", "work_type",
        "Residence_type", "smoking_status",
        "hypertension", "heart_disease"
    ]
    for c in cat_cols:
        df[c] = df[c].astype(object)

    return df

@app.post("/predict")
def predict(p: PredictRequest):
    try:
        df = build_dataframe(p)

        prob = float(pipeline.predict_proba(df)[0][1])
        risk_score = round(prob * 100, 2)

        if risk_score >= 60:
            level = "HIGH"
        elif risk_score >= 30:
            level = "MEDIUM"
        else:
            level = "LOW"

        return {
            "riskLevel": level,
            "probability": round(prob, 4),
            "riskScore": risk_score
        }

    except Exception as e:
        print("==== INFERENCE ERROR ====")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.get("/health")
def health():
    return {"status": "ok", "model_path": str(MODEL_PATH)}
# ===================================================================


# ===================== CHATBOT PART (ADDED) =====================
load_dotenv()  # đọc file .env

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("⚠️ WARNING: GEMINI_API_KEY not found in environment")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        response = gemini_model.generate_content(req.message)
        return {"reply": response.text}
    except Exception as e:
        print("==== CHATBOT ERROR ====")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Chatbot failed: {str(e)}")

@app.get("/chat/health")
def chat_health():
    return {
        "status": "ok",
        "provider": "gemini",
        "model": "gemini-1.5-flash",
        "has_api_key": bool(GEMINI_API_KEY)
    }
# ===============================================================
