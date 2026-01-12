"""
KrashiMitra - AI-Powered Soil Analysis and Crop Recommendation System
Flask Backend Server
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import traceback

from fusion_ai.soil_predictor import SoilPredictor
from fusion_ai.salinity_detector import detect_salinity
from fusion_ai.questionnaire import parse_questionnaire
from fusion_ai.fusion_engine import fuse_all

app = Flask(__name__, static_folder='frontend', static_url_path='')

# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})

# Configuration
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize soil predictor (load model once at startup)
print("Loading AI model...")
soil_predictor = SoilPredictor()
print("Model loaded successfully!")

# In-memory storage for the latest result (for demo purposes)
LAST_ANALYSIS = {}


@app.route('/')
def index():
    """Serve the main index.html page"""
    return send_from_directory('frontend', 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files from frontend directory"""
    if os.path.exists(os.path.join('frontend', path)):
        return send_from_directory('frontend', path)
    return "File not found", 404


@app.route('/analyze', methods=['POST'])
def analyze_soil():
    """
    Main endpoint for soil analysis
    Receives: multipart/form-data with 'image' file and 'answers' JSON string
    Returns: JSON with analysis results
    """
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        # Get questionnaire answers
        answers_str = request.form.get('answers', '{}')
        try:
            answers_json = json.loads(answers_str)
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid JSON in answers"}), 400

        # Save uploaded image
        image_path = os.path.join(UPLOAD_DIR, image_file.filename)
        image_file.save(image_path)
        print(f"Image saved to: {image_path}")

        # 1️⃣ Predict soil type from image using CNN model
        print("Predicting soil type...")
        soil_type = soil_predictor.predict(image_path)
        print(f"Soil type detected: {soil_type}")

        # 2️⃣ Detect salinity and white crust from image
        print("Detecting salinity...")
        salinity, white_ratio = detect_salinity(image_path)
        print(f"Salinity: {salinity}, White ratio: {white_ratio:.2%}")

        # 3️⃣ Process user questionnaire
        print("Processing questionnaire...")
        user_data = parse_questionnaire(answers_json)
        print(f"User data: {user_data}")

        # 4️⃣ Run fusion AI engine to generate final report
        print("Running fusion engine...")
        final_report = fuse_all(
            soil_type=soil_type,
            salinity=salinity,
            white_ratio=white_ratio,
            questionnaire=user_data
        )
        
        print("Analysis complete!")
        
        # Store result in memory
        global LAST_ANALYSIS
        LAST_ANALYSIS = final_report

        return jsonify(final_report), 200

    except Exception as e:
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({
            "error": "Analysis failed",
            "message": error_msg,
            "type": type(e).__name__
        }), 500


@app.route('/result', methods=['GET'])
def get_result():
    """
    Get the last analysis result
    Returns: JSON with last analysis data
    """
    if not LAST_ANALYSIS:
        return jsonify({"error": "No analysis performed yet"}), 404
    return jsonify(LAST_ANALYSIS), 200


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": soil_predictor.model is not None,
        "upload_dir": UPLOAD_DIR
    }), 200


if __name__ == '__main__':
    print("="*60)
    print("🌱 KrashiMitra - AI Soil Analysis System")
    print("="*60)
    print(f"✓ Upload directory: {os.path.abspath(UPLOAD_DIR)}")
    print(f"✓ Frontend directory: {os.path.abspath('frontend')}")
    print(f"✓ Model loaded: {soil_predictor.model is not None}")
    print("="*60)
    print("Starting server at http://127.0.0.1:5000")
    print("Open your browser and navigate to: http://127.0.0.1:5000")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
