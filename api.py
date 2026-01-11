from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import json

from fusion_ai.soil_predictor import predict_soil
from fusion_ai.salinity_detector import detect_salinity
from fusion_ai.questionnaire import parse_questionnaire
from fusion_ai.fusion_engine import fuse_all

app = FastAPI()

# Allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory storage for the latest result (for demo purposes)
LAST_ANALYSIS = {}

@app.post("/analyze")
async def analyze_soil(
    image: UploadFile = File(...),
    answers: str = Form(...)
):
    # Save image
    image_path = os.path.join(UPLOAD_DIR, image.filename)
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # Convert answers JSON
    try:
        answers_json = json.loads(answers)
    except json.JSONDecodeError:
        answers_json = {}

    # 1️⃣ Soil type from image
    soil_type = predict_soil(image_path)

    # 2️⃣ Salinity / white crust from image
    salinity, white_ratio = detect_salinity(image_path)

    # 3️⃣ User questionnaire processing
    user_data = parse_questionnaire(answers_json)

    # 4️⃣ Fusion AI
    final_report = fuse_all(
        soil_type=soil_type,
        salinity=salinity,
        white_ratio=white_ratio,
        questionnaire=user_data
    )
    
    # Store result
    global LAST_ANALYSIS
    LAST_ANALYSIS = final_report

    return final_report

@app.get("/result")
def get_result():
    return LAST_ANALYSIS
