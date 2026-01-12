# 🌱 KrashiMitra – AI-Powered Soil Health & Crop Recommendation System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

KrashiMitra is an intelligent agriculture assistant that empowers farmers with AI-driven soil health analysis and crop recommendations. By combining computer vision, machine learning, and domain expertise, it provides instant insights to improve agricultural productivity.

---

## 📋 Table of Contents

- [Features](#-features)
- [How It Works](#-how-it-works)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Running the Application](#-running-the-application)
- [API Documentation](#-api-documentation)
- [AI Pipeline Details](#-ai-pipeline-details)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

---

## ✨ Features

### 🎯 Core Capabilities
- **🖼️ Image-based Soil Analysis**: Upload soil photos for instant AI classification
- **🧪 Salinity Detection**: Automatic white salt crust detection and quantification
- **📊 Smart Questionnaire**: 11 targeted questions about soil conditions
- **🌾 Crop Recommendations**: Season and condition-aware crop suggestions
- **📈 Health Score**: Overall soil health assessment (0-100)
- **📱 Responsive UI**: Works on desktop, tablet, and mobile devices

### 🔬 Analysis Components
- **Soil Type Classification**: Sandy, Loamy, or Clay (CNN-based)
- **Moisture Analysis**: Water retention and drainage assessment
- **Root Health**: Detection of hardpan and root restrictions
- **Stress Indicators**: Identifies problem areas affecting yield

---

## 🔄 How It Works

### User Journey
```
1. Farmer opens web app (http://127.0.0.1:5000)
2. Answers 11 simple questions about their soil
3. Uploads a photo of the soil
4. AI analyzes the image + answers
5. Receives instant report with:
   ✓ Soil type
   ✓ Salinity level
   ✓ Root health status
   ✓ Top 5 crop recommendations
```

### AI Fusion Process
```
┌─────────────┐      ┌──────────────┐      ┌───────────────┐
│   Soil      │─────→│   CNN Model  │─────→│  Soil Type    │
│   Image     │      │  (Keras)     │      │  Prediction   │
└─────────────┘      └──────────────┘      └───────────────┘
                              │
                              ↓
┌─────────────┐      ┌──────────────┐      ┌───────────────┐
│ Questionnaire│─────→│ Rule Engine  │─────→│   Fusion AI   │
│  (11 Qs)    │      │              │      │   Engine      │
└─────────────┘      └──────────────┘      └───────┬───────┘
                                                    │
                              ↓                     ↓
                     ┌──────────────────────────────────┐
                     │   Final Soil Health Report       │
                     │   + Crop Recommendations         │
                     └──────────────────────────────────┘
```

---

## 🛠️ Tech Stack

### Backend
- **Python 3.8+**: Core programming language
- **Flask 2.3+**: Web framework and REST API
- **TensorFlow 2.13+**: Deep learning model inference
- **OpenCV**: Image processing and salinity detection
- **NumPy**: Numerical computations

### Frontend
- **HTML5 + CSS3**: Structure and styling
- **Tailwind CSS**: Utility-first styling framework
- **Vanilla JavaScript**: Interactive functionality
- **LocalStorage**: Client-side data persistence

### AI/ML
- **Keras CNN Model**: Soil type classification (224×224 RGB input)
- **Computer Vision**: White pixel detection for salinity
- **Rule-based Engine**: Questionnaire analysis and crop mapping

---

## 📁 Project Structure

```
KrashiMitra/
│
├── app.py                          # 🚀 Flask main server (START HERE)
├── api.py                          # (Legacy FastAPI - not used)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── uploads/                        # 📁 User-uploaded soil images
│
├── models/
│   └── soil_classifier.keras       # 🧠 Pre-trained CNN model (224×224×3)
│
├── fusion_ai/                      # 🤖 AI Logic Modules
│   ├── __init__.py
│   ├── fusion_engine.py            # Orchestrates all AI components
│   ├── soil_predictor.py           # CNN model wrapper with fallback
│   ├── salinity_detector.py        # White crust detection (OpenCV)
│   ├── crop_logic.py               # Crop recommendation rules
│   └── questionnaire.py            # Parses farmer responses
│
├── frontend/                       # 🎨 User Interface
│   ├── index.html                  # Landing page
│   ├── question.html               # 11-question form
│   ├── upload.html                 # Image upload page
│   ├── report.html                 # Analysis results display
│   ├── config.js                   # API endpoint configuration
│   └── data/
│       └── questions.json          # (Optional) Question metadata
│
└── assests/                        # Static assets (images, etc.)
    └── images/
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- 500 MB free disk space (for dependencies)

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd KrashiMitra
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv env
env\Scripts\activate

# macOS/Linux
python3 -m venv env
source env/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Installation time**: ~5-10 minutes (depending on internet speed)

### Step 4: Verify Model File
Ensure `models/soil_classifier.keras` exists. If missing:
- The system will use a fallback color-based predictor
- Or download/train a new model and place it in `models/`

---

## 🏃 Running the Application

### Start the Server
```bash
python app.py
```

**Expected output:**
```
============================================================
🌱 KrashiMitra - AI Soil Analysis System
============================================================
Loading AI model...
✓ Soil classification model loaded from models/soil_classifier.keras
✓ Upload directory: C:\...\KrashiMitra\uploads
✓ Frontend directory: C:\...\KrashiMitra\frontend
✓ Model loaded: True
============================================================
Starting server at http://127.0.0.1:5000
Open your browser and navigate to: http://127.0.0.1:5000
============================================================
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://<your-ip>:5000
```

### Access the Application
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

or

```
http://localhost:5000
```

### Using the Application

1. **Home Page**: Click "Start Soil Test" or "Check Soil"
2. **Questionnaire**: Answer 11 questions about soil conditions
3. **Upload Image**: Take/upload a clear photo of your soil
4. **View Report**: Get instant analysis and crop recommendations

---

## 📡 API Documentation

### Base URL
```
http://localhost:5000
```

### Endpoints

#### 1. Analyze Soil
**POST** `/analyze`

Processes soil image and questionnaire data to generate analysis report.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body:
  - `image`: Image file (JPG, PNG, max 5MB)
  - `answers`: JSON string with questionnaire responses

**Example Request (JavaScript):**
```javascript
const formData = new FormData();
formData.append('image', imageFile);
formData.append('answers', JSON.stringify({
    season: "Kharif",
    crop: "Wheat",
    moisture: "Moist",
    texture: "Loamy",
    cracks: "None",
    absorption: "Fast",
    crust: "No",
    root_layer: "shallow",
    yield: "Good"
}));

fetch('http://localhost:5000/analyze', {
    method: 'POST',
    body: formData
})
.then(res => res.json())
.then(data => console.log(data));
```

**Response (200 OK):**
```json
{
    "soil": "Loamy",
    "salinity": "low",
    "white_ratio": 0.0234,
    "root_condition": "healthy",
    "moisture_level": "Moist",
    "season": "Kharif",
    "recommended_crops": ["Rice", "Maize", "Cotton", "Soybean", "Groundnut"],
    "health_score": 95
}
```

**Error Response (500):**
```json
{
    "error": "Analysis failed",
    "message": "Image file corrupted",
    "type": "IOError"
}
```

---

#### 2. Get Last Result
**GET** `/result`

Retrieves the most recent analysis result.

**Response (200 OK):**
```json
{
    "soil": "Sandy",
    "salinity": "medium",
    "white_ratio": 0.0876,
    "root_condition": "restricted",
    "moisture_level": "Dry",
    "season": "Rabi",
    "recommended_crops": ["Millet", "Groundnut", "Watermelon"],
    "health_score": 72
}
```

---

#### 3. Health Check
**GET** `/health`

Checks if the server and model are running correctly.

**Response (200 OK):**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "upload_dir": "uploads"
}
```

---

#### 4. Serve Frontend
**GET** `/` or `/<path>`

Serves static HTML files from the `frontend/` directory.

---

## 🧠 AI Pipeline Details

### 1. Soil Type Classification

**Model**: Convolutional Neural Network (CNN)
- **Architecture**: Custom trained Keras model
- **Input**: 224×224×3 RGB images
- **Output**: 3 classes (Sandy, Loamy, Clay)
- **Preprocessing**: Resize + Normalization (0-1 scale)

**Fallback Logic**: If model fails, uses color-based heuristics:
- Light/yellowish → Sandy
- Dark/reddish → Clay
- Medium tone → Loamy

### 2. Salinity Detection

**Algorithm**: White pixel threshold detection
```python
1. Convert image to grayscale
2. Count pixels with intensity > 200
3. Calculate ratio = white_pixels / total_pixels
4. Classify:
   - ratio > 15% → High salinity
   - ratio > 5%  → Medium salinity
   - ratio ≤ 5%  → Low salinity
```

### 3. Crop Recommendation Logic

**Factors Considered**:
- Soil type (Sandy/Loamy/Clay)
- Moisture level (Dry/Moist/Wet)
- Salinity (Low/Medium/High)
- Season (Rabi/Kharif/Zaid)

**Priority Rules**:
1. High salinity → Only salt-tolerant crops (Barley, Cotton)
2. Medium salinity → Moderately tolerant crops
3. Low salinity → Full range based on soil + season

**Crop Database**:
- **Rabi (Winter)**: Wheat, Barley, Gram, Mustard, Peas
- **Kharif (Monsoon)**: Rice, Maize, Cotton, Soybean, Groundnut
- **Zaid (Summer)**: Watermelon, Cucumber, Muskmelon, Vegetables

### 4. Health Score Calculation

```python
Base Score: 100
Deductions:
- High salinity: -30 points
- Medium salinity: -15 points
- Restricted roots: -20 points
- White crust: -(white_ratio × 100) points

Final Score: max(0, min(100, adjusted_score))
```

---

## 🐛 Troubleshooting

### Issue 1: Server Won't Start
**Error**: `ModuleNotFoundError: No module named 'flask'`

**Solution**:
```bash
pip install -r requirements.txt
```

---

### Issue 2: Model File Missing
**Error**: `Model file not found at: models/soil_classifier.keras`

**Solution**:
- App will use fallback predictor (color-based)
- Or download a trained model and place it in `models/`

---

### Issue 3: Frontend Not Loading
**Error**: Blank page at `http://localhost:5000`

**Checks**:
1. Verify `frontend/` directory exists
2. Check console for errors: `F12` → Console tab
3. Ensure `app.py` is running (not `api.py`)

**Solution**:
```bash
# Restart server
python app.py
```

---

### Issue 4: CORS Error in Browser
**Error**: `Access-Control-Allow-Origin`

**Solution**: CORS is already configured in `app.py`. Ensure:
- Using `http://localhost:5000` (not file:// protocol)
- Browser cache cleared (`Ctrl+F5`)

---

### Issue 5: Image Upload Fails
**Error**: `413 Request Entity Too Large`

**Solution**: Image size > 5MB. Compress image or update limit in code.

---

### Issue 6: Wrong Port
**Problem**: Server runs on port 8000 instead of 5000

**Solution**: Update `config.js`:
```javascript
const API_BASE_URL = "http://localhost:5000/analyze";
```

---

## 📊 Performance Notes

- **Model Inference**: ~0.5-2 seconds per image (CPU)
- **Salinity Detection**: ~0.1-0.3 seconds
- **API Response Time**: ~1-3 seconds total
- **Memory Usage**: ~500MB (with TensorFlow loaded)

---

## 🔐 Security Considerations

⚠️ **This is a demonstration/educational project**. For production:
- [ ] Add authentication and authorization
- [ ] Validate and sanitize all inputs
- [ ] Implement rate limiting
- [ ] Use HTTPS
- [ ] Store images securely
- [ ] Add logging and monitoring

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` file for details.

---

## 👥 Authors

**KrashiMitra Development Team**
- AI/ML Engineers
- Full-Stack Developers
- Agricultural Domain Experts

---

## 🙏 Acknowledgments

- TensorFlow and Keras teams for ML frameworks
- OpenCV community for computer vision tools
- Agricultural research community for domain knowledge
- Flask framework developers

---

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Email: support@krashimitra.example.com
- Documentation: [Wiki](https://github.com/yourrepo/wiki)

---

**Made with ❤️ for farmers and agriculture**

### 3. Run the Frontend
You can simply open `frontend/index.html` in your browser.
or use Live Server / Python HTTP server:

```sh
cd frontend
python -m http.server 5500
```
Then open `http://localhost:5500` in your browser.
