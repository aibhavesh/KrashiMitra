import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("models/soil_classifier.keras")

def predict_soil(image_path):
    img = Image.open(image_path).resize((224,224))
    img = np.array(img)/255.0
    img = img.reshape(1,224,224,3)

    pred = model.predict(img)[0]
    labels = ["sandy", "loamy", "clay"]

    return labels[np.argmax(pred)]
