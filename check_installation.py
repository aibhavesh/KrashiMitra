"""
Quick Start Script for KrashiMitra
Run this to verify your installation
"""
import sys
import subprocess

def check_python_version():
    """Check if Python version is 3.8+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. You have:", sys.version)
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required = [
        'flask',
        'flask_cors',
        'tensorflow',
        'cv2',
        'numpy',
        'PIL'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            missing.append(package)
            print(f"❌ {package} missing")
    
    return len(missing) == 0, missing

def check_model_file():
    """Check if model file exists"""
    import os
    if os.path.exists('models/soil_classifier.keras'):
        print("✓ Model file found")
        return True
    else:
        print("⚠ Model file missing (will use fallback)")
        return False

def check_frontend():
    """Check if frontend files exist"""
    import os
    files = ['index.html', 'question.html', 'upload.html', 'report.html']
    all_exist = True
    for file in files:
        path = f'frontend/{file}'
        if os.path.exists(path):
            print(f"✓ {file}")
        else:
            print(f"❌ {file} missing")
            all_exist = False
    return all_exist

def main():
    print("="*60)
    print("🌱 KrashiMitra Installation Checker")
    print("="*60)
    
    print("\n1. Checking Python version...")
    if not check_python_version():
        return
    
    print("\n2. Checking dependencies...")
    deps_ok, missing = check_dependencies()
    if not deps_ok:
        print("\n❌ Missing packages. Install with:")
        print("   pip install -r requirements.txt")
        return
    
    print("\n3. Checking model file...")
    check_model_file()
    
    print("\n4. Checking frontend files...")
    if not check_frontend():
        print("\n❌ Frontend files missing!")
        return
    
    print("\n" + "="*60)
    print("✓ All checks passed! Ready to run.")
    print("="*60)
    print("\nTo start the server, run:")
    print("   python app.py")
    print("\nThen open: http://127.0.0.1:5000")

if __name__ == "__main__":
    main()
