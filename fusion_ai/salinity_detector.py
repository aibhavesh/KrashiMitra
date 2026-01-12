"""
Salinity Detector - Analyzes white salt crust in soil images
"""
import cv2
import numpy as np
import os


def detect_salinity(image_path):
    """
    Detect salinity level by analyzing white pixels in soil image
    
    Args:
        image_path: Path to the soil image
        
    Returns:
        tuple: (salinity_level, white_ratio)
            - salinity_level: 'low', 'medium', or 'high'
            - white_ratio: float percentage of white pixels (0.0 to 1.0)
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Unable to read image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Count white/bright pixels (threshold at 200)
        white_pixels = np.sum(gray > 200)
        total_pixels = gray.size
        
        # Calculate ratio
        ratio = white_pixels / total_pixels if total_pixels > 0 else 0.0
        
        # Classify salinity level
        if ratio > 0.15:
            salinity_level = "high"
        elif ratio > 0.05:
            salinity_level = "medium"
        else:
            salinity_level = "low"
        
        print(f"  Salinity analysis: {salinity_level} (white ratio: {ratio:.2%})")
        return salinity_level, ratio
        
    except Exception as e:
        print(f"Error in salinity detection: {e}")
        # Return safe default
        return "low", 0.0
