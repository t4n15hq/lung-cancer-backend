import os
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
import numpy as np
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
import time
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inline ClinicalFeatureExtractor (your existing class)
class ClinicalFeatureExtractor:
    """Extract clinical features from patient data, using only provided values."""
    def __init__(self):
        self.feature_names = []

    def create_clinical_features(self, patient_data: dict) -> dict:
        age = patient_data["age"]
        sex = patient_data["sex"]
        smoking = patient_data["smoking_pack_years"]
        cea = patient_data["cea_level"]
        hgb = patient_data["hemoglobin"]
        wbc = patient_data["wbc_count"]
        weight_loss = patient_data["weight_loss"]
        cough = patient_data["persistent_cough"]
        dyspnea = patient_data["dyspnea"]
        chest_pain = patient_data["chest_pain"]

        features = {
            # Demographics
            "age": age,
            "age_squared": age**2,
            "age_over_65": int(age > 65),
            "age_over_75": int(age > 75),
            "male": int(sex == "M"),
            # Smoking
            "smoking_pack_years": smoking,
            "smoking_binary": int(smoking > 0),
            "moderate_smoker": int(10 <= smoking <= 30),
            "heavy_smoker": int(smoking > 30),
            "very_heavy_smoker": int(smoking > 50),
            "smoking_age_interaction": smoking * age / 1000,
            # Laboratory values
            "cea_level_raw": cea,
            "cea_level_log": np.log1p(cea),
            "cea_mildly_elevated": int(3.0 < cea <= 10.0),
            "cea_elevated": int(cea > 3.0),
            "cea_very_high": int(cea > 10.0),
            "cea_extremely_high": int(cea > 20.0),
            # Hemoglobin
            "hemoglobin_level": hgb,
            "mild_anemia": int(10.0 <= hgb < 12.0),
            "anemia": int(hgb < 12.0),
            "severe_anemia": int(hgb < 10.0),
            # White blood cells
            "wbc_count_normalized": wbc / 1000,
            "leukopenia": int(wbc < 4000),
            "leukocytosis": int(wbc > 11000),
            "marked_leukocytosis": int(wbc > 15000),
            # Symptoms
            "weight_loss": weight_loss,
            "persistent_cough": cough,
            "dyspnea": dyspnea,
            "chest_pain": chest_pain,
            "constitutional_symptoms": weight_loss,
            "respiratory_symptoms": max(cough, dyspnea),
            "cough_and_weight_loss": weight_loss * cough,
            "dyspnea_and_weight_loss": weight_loss * dyspnea,
        }

        # Composite scores
        features["symptom_count"] = (
            features["weight_loss"]
            + features["persistent_cough"]
            + features["dyspnea"]
            + features["chest_pain"]
        )
        features["multiple_symptoms"] = int(features["symptom_count"] >= 2)
        features["many_symptoms"] = int(features["symptom_count"] >= 3)

        features["lab_abnormal_count"] = (
            features["cea_elevated"]
            + features["anemia"]
            + features["leukocytosis"]
        )
        features["multiple_lab_abnormalities"] = int(
            features["lab_abnormal_count"] >= 2
        )

        features["lung_cancer_risk_score"] = (
            features["age_over_65"] * 0.2
            + features["heavy_smoker"] * 0.3
            + features["cea_elevated"] * 0.3
            + features["constitutional_symptoms"] * 0.2
        )
        features["high_risk_profile"] = int(
            features["age_over_65"]
            and features["heavy_smoker"]
            and features["cea_elevated"]
            and features["constitutional_symptoms"]
        )

        if not self.feature_names:
            self.feature_names = list(features.keys())

        return features

# Inline EnhancedTextProcessor (your existing class)
class EnhancedTextProcessor:
    """Enhanced text processing with better bias removal."""
    def __init__(self):
        self.research_stopwords = [
            'study', 'studies', 'research', 'analysis', 'findings', 'results',
            'conclusion', 'objective', 'methods', 'background', 'trial',
            'systematic', 'review', 'meta', 'cohort', 'retrospective',
            'database', 'median', 'mean', 'significant', 'correlation',
            'participants', 'measurements', 'outcome', 'design', 'setting',
            'prospective', 'randomized', 'controlled', 'multicenter',
            'investigation', 'evaluated', 'assessed', 'examined'
        ]

    def clean_medical_text_enhanced(self, text: str) -> str:
        text = text.lower()
        patterns = [
            r'\b(?:this|our|the present)\s+(?:study|research|analysis|investigation)\b',
            r'\b(?:studies|research)\s+(?:showed?|demonstrated?|found|revealed?|indicated?)\b',
            r'\bmeta-analysis\b', r'\bsystematic\s+review\b',
            r'\bliterature\s+review\b', r'\brandomized\s+controlled\s+trial\b',
            r'\bprospective\s+study\b', r'\bretro?spective\s+(?:study|analysis)\b'
        ]
        for p in patterns:
            text = re.sub(p, ' ', text)
        text = re.sub(r'http[s]?://\S+', ' ', text)
        text = re.sub(r'\[\d+\]', ' ', text)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def create_enhanced_vectorizer(self, max_features: int = 3000):
        stops = list(ENGLISH_STOP_WORDS) + self.research_stopwords
        return TfidfVectorizer(
            max_features=max_features,
            stop_words=stops,
            ngram_range=(1, 3),
            min_df=5,
            max_df=0.7,
            sublinear_tf=True,
            norm='l2',
            token_pattern=r'\b[a-z]{3,}\b',
            lowercase=True,
            strip_accents='ascii'
        )

# Custom unpickler to map classes
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "ClinicalFeatureExtractor":
            return ClinicalFeatureExtractor
        if module == "__main__" and name == "EnhancedTextProcessor":
            return EnhancedTextProcessor
        return super().find_class(module, name)

# Schemas
class PatientData(BaseModel):
    age: int
    sex: str
    smoking_pack_years: float
    cea_level: float
    hemoglobin: float
    wbc_count: float
    weight_loss: int
    persistent_cough: int
    dyspnea: int
    chest_pain: int

class TextData(BaseModel):
    text: str

# Global model storage
ml_models = {}

def load_model():
    """Load the ML model with error handling and logging."""
    global ml_models
    
    try:
        model_path = "robust_lung_cancer_model.pkl"
        
        # Check if model file exists
        if not os.path.exists(model_path):
            logger.error(f"Model file {model_path} not found!")
            raise FileNotFoundError(f"Model file {model_path} not found")
        
        logger.info(f"Loading model from {model_path}...")
        
        with open(model_path, "rb") as f:
            data = CustomUnpickler(f).load()

        # Store all model components
        ml_models.update({
            "feature_extractor": data["feature_extractor"],
            "feature_names": data["feature_names"],
            "clinical_scaler": data["clinical_scaler"],
            "clinical_model": data["clinical_model"],
            "text_processor": data["text_processor"],
            "text_vectorizer": data["text_vectorizer"],
            "text_model": data["text_model"]
        })

        logger.info("✅ Model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        return False

# Lifespan events for proper startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting Lung Cancer Risk Prediction API...")
    success = load_model()
    if not success:
        logger.error("❌ Failed to start: Model loading failed")
        raise RuntimeError("Model loading failed")
    
    logger.info("✅ API ready to serve predictions!")
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down API...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Lung Cancer Risk Prediction API",
    description="AI-powered lung cancer risk assessment using clinical data and text analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"]  # Configure this for production
)

# CORS middleware - configure for your frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "https://your-frontend-domain.com",  # Your production domain
        "*"  # Remove this in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Health endpoints
@app.get("/")
async def read_root():
    return {
        "status": "running",
        "service": "Lung Cancer Risk Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    model_status = "loaded" if ml_models else "not_loaded"
    
    return {
        "status": "healthy" if ml_models else "unhealthy",
        "model_status": model_status,
        "timestamp": time.time(),
        "service": "lung_cancer_api"
    }

@app.get("/model/status")
async def model_status():
    """Detailed model status"""
    if not ml_models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "clinical_model": "loaded" if ml_models.get("clinical_model") else "missing",
        "text_model": "loaded" if ml_models.get("text_model") else "missing",
        "feature_extractor": "loaded" if ml_models.get("feature_extractor") else "missing",
        "vectorizer": "loaded" if ml_models.get("text_vectorizer") else "missing",
        "total_components": len(ml_models)
    }

# Prediction endpoints
@app.post("/predict")
async def predict_clinical(payload: PatientData):
    """Predict lung cancer risk from clinical data"""
    if not ml_models.get("clinical_model"):
        raise HTTPException(status_code=503, detail="Clinical model not loaded")

    try:
        # Extract features
        feats = ml_models["feature_extractor"].create_clinical_features(payload.dict())
        
        # Prepare feature array
        arr = np.array([feats[f] for f in ml_models["feature_names"]]).reshape(1, -1)
        
        # Scale and predict
        Xs = ml_models["clinical_scaler"].transform(arr)
        prob = ml_models["clinical_model"].predict_proba(Xs)[0][1]
        pred = int(prob >= 0.5)
        
        logger.info(f"Clinical prediction: {pred}, probability: {prob:.3f}")
        
        return {
            "prediction": pred,
            "probability": float(prob),
            "risk_level": "high" if prob >= 0.5 else "low",
            "confidence": float(prob) if pred == 1 else float(1 - prob)
        }
        
    except Exception as e:
        logger.error(f"Clinical prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_text")
async def predict_text(payload: TextData):
    """Predict lung cancer risk from text"""
    if not ml_models.get("text_model"):
        raise HTTPException(status_code=503, detail="Text model not loaded")

    try:
        # Clean and vectorize text
        cleaned = ml_models["text_processor"].clean_medical_text_enhanced(payload.text)
        
        if not cleaned.strip():
            raise HTTPException(status_code=400, detail="No valid text content after cleaning")
        
        Xt = ml_models["text_vectorizer"].transform([cleaned])
        prob = ml_models["text_model"].predict_proba(Xt)[0][1]
        pred = int(prob >= 0.5)
        
        logger.info(f"Text prediction: {pred}, probability: {prob:.3f}")
        
        return {
            "prediction": pred,
            "probability": float(prob),
            "risk_level": "high" if prob >= 0.5 else "low",
            "confidence": float(prob) if pred == 1 else float(1 - prob),
            "processed_text_length": len(cleaned)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text prediction failed: {str(e)}")

# For local development
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "server:app",
        host="0.0.0.0", 
        port=port,
        reload=False,  # Set to False for production
        access_log=True
    )