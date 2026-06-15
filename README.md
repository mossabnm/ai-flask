# KidneyVision AI - Flask Microservice

This microservice exposes the trained Keras model (`best_model.keras`) via a simple REST API.

## Requirements
- Python 3.10+
- TensorFlow 2.15+

## Running Locally (Windows)
1. Open PowerShell in this directory.
2. Create a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Run the server:
   ```powershell
   python app.py
   ```

## Running with Docker
```bash
docker build -t kidneyvision-ai .
docker run -p 5000:5000 kidneyvision-ai
```

## API Endpoints

### GET /health
Returns the health status of the API and the loaded model.
```json
{
  "status": "healthy",
  "model_version": "best_model.keras"
}
```

### POST /predict
Predicts whether a kidney ultrasound contains a stone.
**Headers:** `Content-Type: multipart/form-data`
**Body:** `image` (File)

**Response (Success):**
```json
{
  "prediction": "Stone",
  "confidence": 94.35,
  "probability": 0.9435,
  "model_version": "best_model.keras"
}
```
