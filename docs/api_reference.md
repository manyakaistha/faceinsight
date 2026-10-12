# API Reference

Base URL: http://localhost:8000

Endpoints:
- GET /api/v1/health
- GET /api/v1/ready
- GET /api/v1/model/info
- POST /api/v1/predict (multipart form-data, file=image/*) -> returns faces with bbox, age, gender, prob, threshold
- POST /api/v1/predict/batch (multiple files, max 32) -> list
- GET /api/v1/config/public
- GET /api/v1/metrics
