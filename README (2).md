# GreenArk YOLOv8 Render API

This service deploys the supplied YOLOv8n classification model as an HTTP API using ONNX Runtime.

## Classes
- metal
- other
- plastic

## Local test

```bash
pip install -r requirements.txt
python app.py
```

Then open:
- http://localhost:10000/health

Test inference:

```bash
curl -X POST -F "file=@plastic.jpg" http://localhost:10000/predict
```

## Render

1. Create a GitHub repository.
2. Put all files in this folder at the repository root.
3. Push to GitHub.
4. In Render, create a new Web Service and connect the repository.
5. Build command:
   `pip install -r requirements.txt`
6. Start command:
   `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120`
7. Deploy.
8. Test:
   `https://YOUR-SERVICE.onrender.com/health`

Prediction endpoint:

`POST https://YOUR-SERVICE.onrender.com/predict`

Send an image as multipart/form-data using the field name `file`.

## GreenArk / ESP32-CAM request

The ESP32-CAM should capture a JPEG and send:

```http
POST /predict
Content-Type: multipart/form-data
```

with the JPEG under form field `file`.

The response contains:

```json
{
  "success": true,
  "class": "plastic",
  "predicted_class": "plastic",
  "confidence": 0.96,
  "margin": 0.82,
  "accepted": true,
  "thresholds": {
    "confidence": 0.8,
    "margin": 0.15
  },
  "probabilities": {
    "metal": 0.02,
    "other": 0.02,
    "plastic": 0.96
  }
}
```

## Important

The model is a classifier, not an object detector. It classifies the image as one of the three classes; it does not return bounding boxes.

Render cold starts and network latency can still affect first-request response time on a free/sleeping instance.
