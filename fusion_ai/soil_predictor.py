"""
Soil Type Predictor - Uses CNN model to classify soil from images
"""
import tensorflow as tf
import numpy as np
from PIL import Image
import os


class SoilPredictor:
    """Soil classification using trained Keras CNN model"""
    
    def __init__(self, model_path="models/soil_classifier.keras"):
        """
        Initialize soil predictor and load model
        
        Args:
            model_path: Path to the trained Keras model file
        """
        self.model_path = model_path
        self.model = None
        self.labels = ["sandy", "loamy", "clay"]
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the Keras model from disk"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at: {self.model_path}")
            
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"✓ Soil classification model loaded from {self.model_path}")
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print("⚠ Using fallback dummy predictor")
            self.model = None
    
    def predict(self, image_path):
        """
        Predict soil type from an image
        
        Args:
            image_path: Path to the soil image file
            
        Returns:
            str: Predicted soil type ('sandy', 'loamy', or 'clay')
        """
        try:
            # If model failed to load, use fallback
            if self.model is None:
                return self._fallback_predict(image_path)
            
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB').resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = img_array.reshape(1, 224, 224, 3)
            
            # Predict
            predictions = self.model.predict(img_array, verbose=0)[0]
            predicted_idx = np.argmax(predictions)
            
            return self.labels[predicted_idx]
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return self._fallback_predict(image_path)
    
    def _fallback_predict(self, image_path):
        """
        Fallback prediction based on simple image analysis
        Used when CNN model is not available
        """
        try:
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)
            
            # Simple heuristic: analyze color distribution
            mean_color = img_array.mean(axis=(0, 1))
            r, g, b = mean_color
            
            # Sandy soil tends to be lighter/yellowish
            if r > 150 and g > 130:
                return "sandy"
            # Clay soil tends to be darker/reddish
            elif r > g and r > b:
                return "clay"
            # Default to loamy
            else:
                return "loamy"
                
        except:
            # Ultimate fallback
            return "loamy"


# Legacy function for backward compatibility
def predict_soil(image_path):
    """
    Legacy function - creates a new predictor instance
    Use SoilPredictor class instead for better performance
    """
    predictor = SoilPredictor()
    return predictor.predict(image_path)
