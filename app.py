from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from io import BytesIO
from PIL import Image
import sqlite3
import os

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
import gdown

MODEL_PATH = "mango_disease_model.h5"
MODEL_URL = "https://drive.google.com/1l9gI3iQROToXICjB-y_GMRZsp3GJvmjD"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file {MODEL_PATH} not found. Train and save the model first.")
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels
CLASS_LABELS = ["Anthracnose", "Bacterial Canker", "Cutting Weevil", "Die Back", "Gall Midge", "Healthy", "Powdery Mildew", "Sooty Mould"]

# Cure recommendations
CURE_RECOMMENDATIONS = {
    "Anthracnose": "Use copper-based fungicides and avoid overhead watering.",
    "Bacterial Canker": "Apply bactericides and prune infected branches.",
    "Cutting Weevil": "Use insecticides and remove affected leaves.",
    "Die Back": "Apply fungicides and ensure proper soil drainage.",
    "Gall Midge": "Use neem-based pesticides and remove infected twigs.",
    "Healthy": "No treatment needed. Maintain regular monitoring.",
    "Powdery Mildew": "Apply sulfur-based fungicides and ensure proper air circulation.",
    "Sooty Mould": "Control aphids and wash leaves with mild soap solution."
}

# Connect to SQLite database
def init_db():
    conn = sqlite3.connect("mango_disease.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            prediction TEXT,
            cure TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Load image
        contents = await file.read()
        img = Image.open(BytesIO(contents)).resize((150, 150))
        img_array = image.img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        predicted_class = CLASS_LABELS[np.argmax(predictions)]
        cure = CURE_RECOMMENDATIONS[predicted_class]

        # Save prediction to database
        conn = sqlite3.connect("mango_disease.db")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO predictions (filename, prediction, cure) VALUES (?, ?, ?)", 
                       (file.filename, predicted_class, cure))
        conn.commit()
        conn.close()

        return {"filename": file.filename, "prediction": predicted_class, "cure": cure}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the server (for local testing)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
