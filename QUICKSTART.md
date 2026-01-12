# 🚀 Quick Start Guide - KrashiMitra

## For Absolute Beginners

### Windows Users

1. **Install Python**
   - Download from: https://www.python.org/downloads/
   - ✅ Check "Add Python to PATH" during installation
   - Version needed: 3.8 or higher

2. **Run Installation Script**
   ```cmd
   install.bat
   ```
   This will automatically:
   - Create virtual environment
   - Install all dependencies
   - Verify installation

3. **Start the Server**
   ```cmd
   env\Scripts\activate
   python app.py
   ```

4. **Open Browser**
   Navigate to: `http://localhost:5000`

---

### Linux/macOS Users

1. **Install Python** (if not already installed)
   ```bash
   # Ubuntu/Debian
   sudo apt-get install python3 python3-venv python3-pip
   
   # macOS (with Homebrew)
   brew install python3
   ```

2. **Run Installation Script**
   ```bash
   chmod +x install.sh
   ./install.sh
   ```

3. **Start the Server**
   ```bash
   source env/bin/activate
   python app.py
   ```

4. **Open Browser**
   Navigate to: `http://localhost:5000`

---

## Manual Installation (Alternative)

If the automated scripts don't work:

```bash
# Step 1: Create virtual environment
python -m venv env

# Step 2: Activate it
# Windows:
env\Scripts\activate
# Linux/macOS:
source env/bin/activate

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Run the app
python app.py
```

---

## Testing the Installation

Run the verification script:
```bash
python check_installation.py
```

You should see:
```
✓ Python 3.x.x
✓ flask installed
✓ tensorflow installed
✓ cv2 installed
...
✓ All checks passed!
```

---

## Common Issues

### Issue: "python: command not found"
**Solution**: Install Python or use `python3` instead

### Issue: "pip: command not found"
**Solution**: Use `python -m pip` instead of `pip`

### Issue: Port 5000 already in use
**Solution**: Kill the process or change port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=8080)
```

### Issue: Model not found warning
**Solution**: This is OK! The app will use a fallback predictor.

---

## Using the Application

### Step-by-Step Guide

1. **Start the Server** (if not already running)
   ```bash
   python app.py
   ```

2. **Open Your Browser**
   Go to: http://localhost:5000

3. **Click "Start Soil Test"**

4. **Answer 11 Questions**
   - Season (Rabi/Kharif/Zaid)
   - Crop type
   - Water conditions
   - Soil texture
   - etc.

5. **Upload Soil Photo**
   - Take a clear photo of your soil
   - Max size: 5MB
   - Formats: JPG, PNG

6. **View Results**
   You'll see:
   - Soil type (Sandy/Loamy/Clay)
   - Salinity level
   - Root health
   - Recommended crops

---

## API Testing (for Developers)

### Using curl:
```bash
curl -X POST http://localhost:5000/analyze \
  -F "image=@soil_photo.jpg" \
  -F 'answers={"season":"Kharif","crop":"Wheat","moisture":"Moist","texture":"Loamy","cracks":"None","absorption":"Fast","crust":"No","root_layer":"shallow","yield":"Good"}'
```

### Using Postman:
1. Create new POST request
2. URL: `http://localhost:5000/analyze`
3. Body → form-data
4. Add key "image" (type: File)
5. Add key "answers" (type: Text, value: JSON string)
6. Send!

---

## Stopping the Server

Press `Ctrl + C` in the terminal where the server is running.

---

## Getting Help

1. Check README.md for detailed documentation
2. Run `python check_installation.py` to verify setup
3. Look at server logs in terminal for error messages
4. Check browser console (F12) for frontend errors

---

## Project Structure Quick Reference

```
KrashiMitra/
├── app.py              ← START HERE (main server)
├── requirements.txt    ← Dependencies
├── models/             ← AI model files
├── fusion_ai/          ← AI logic
├── frontend/           ← Web interface
└── uploads/            ← Uploaded images
```

---

## Deployment Notes

**For Production:**
- Use Gunicorn/uWSGI instead of Flask dev server
- Set `debug=False` in app.py
- Use HTTPS
- Add authentication
- Set up proper logging
- Use a production database

**Example with Gunicorn:**
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

---

**Made with ❤️ for farmers**
