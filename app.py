import os
import logging
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
import tensorflow as tf

# Initialize Flask App
app = Flask(__name__)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = os.environ.get("MODEL_PATH", "best_model.keras")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
IMG_HEIGHT, IMG_WIDTH = 224, 224
CLASS_NAMES = {0: "Normal", 1: "Stone"}
THRESHOLD = 0.5

model = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_keras_model():
    """Load the model once at startup and warm it up."""
    global model
    try:
        logger.info(f"Loading model from {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded successfully.")
        
        # Warm-up the model to reduce first prediction latency
        logger.info("Warming up the model...")
        dummy_input = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
        model.predict(dummy_input, verbose=0)
        logger.info("Model warm-up complete.")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        # We don't exit here so the health endpoint can report failure if needed
        model = None

# Execute model loading
load_keras_model()

@app.route('/health', methods=['GET'])
def health_check():
    """Health endpoint for the API."""
    if model is None:
        return jsonify({"status": "unhealthy", "message": "Model not loaded"}), 503
    return jsonify({"status": "healthy", "model_version": MODEL_PATH}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint."""
    logger.info("Received prediction request.")
    
    if model is None:
        logger.error("Prediction attempted but model is not loaded.")
        return jsonify({"error": "Service unavailable. Model not loaded."}), 503

    # Check if the post request has the file part
    if 'image' not in request.files:
        logger.warning("No image part in the request.")
        return jsonify({"error": "No image provided"}), 400
        
    file = request.files['image']
    
    if file.filename == '':
        logger.warning("No selected file.")
        return jsonify({"error": "No selected file"}), 400
        
    if not allowed_file(file.filename):
        logger.warning(f"Invalid file extension: {file.filename}")
        return jsonify({"error": f"Invalid file type. Allowed: {ALLOWED_EXTENSIONS}"}), 400

    # Read image
    try:
        image_bytes = file.read()
        if len(image_bytes) > MAX_FILE_SIZE:
            logger.warning(f"File size exceeds limit: {len(image_bytes)} bytes")
            return jsonify({"error": "File exceeds maximum size of 5MB"}), 413
            
        # Re-seek or use BytesIO
        import io
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        logger.error(f"Error reading image: {str(e)}")
        return jsonify({"error": "Invalid image file"}), 400

    # Preprocessing exactly matching training
    try:
        # Convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Resize to 224x224
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        
        # Convert to numpy array and normalize to [0, 1] as expected by most Keras models
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Expand dimensions to (1, 224, 224, 3)
        img_batch = np.expand_dims(img_array, axis=0)
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return jsonify({"error": "Error preprocessing image"}), 500

    # Run inference
    try:
        # Binary prediction probability
        prob = float(model.predict(img_batch, verbose=0)[0][0])
        
        # Map to class
        class_idx = 1 if prob > THRESHOLD else 0
        prediction_label = CLASS_NAMES[class_idx]
        
        # Calculate formatted confidence
        # If it's a binary sigmoid, probability represents the '1' class (Stone).
        # So confidence for the predicted class:
        confidence = prob if class_idx == 1 else (1.0 - prob)
        
        # Format confidence e.g. 0.9435 -> 94.35
        confidence_percent = round(confidence * 100, 2)
        
        logger.info(f"Prediction: {prediction_label}, Probability: {prob:.4f}, Confidence: {confidence_percent}%")
        
        return jsonify({
            "prediction": prediction_label,
            "confidence": confidence_percent,
            "probability": round(prob, 4),
            "model_version": MODEL_PATH
        }), 200

    except Exception as e:
        logger.error(f"Error running inference: {str(e)}")
        return jsonify({"error": "Error during prediction inference"}), 500

if __name__ == '__main__':
    # Used only for local testing, gunicorn will run this in production
    app.run(host='0.0.0.0', port=5000, debug=False)
