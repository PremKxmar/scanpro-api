# API Server

Flask REST API for the document scanner, designed for mobile app integration.

## Quick Start

```bash
# Install Flask
pip install flask flask-cors

# Run the API server
python api/app.py --port 5000
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/scan` | POST | Full document scanning (detect + warp + enhance) |
| `/api/detect` | POST | Detect document corners only |
| `/api/enhance` | POST | Enhance image (shadow removal, sharpening) |
| `/api/export-pdf` | POST | Export images to PDF |
| `/api/health` | GET | Health check |
| `/api/info` | GET | API capabilities |

## Usage Example

### Scan Document (JavaScript/React Native)

```javascript
const scanDocument = async (imageBase64) => {
  const response = await fetch('http://YOUR_SERVER:5000/api/scan', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image: imageBase64,
      options: {
        remove_shadows: true,
        enhance: true,
        output_format: 'jpeg'
      }
    })
  });
  
  const result = await response.json();
  
  if (result.success) {
    console.log('Corners:', result.corners);
    console.log('Confidence:', result.confidence);
    // result.scan contains the scanned image as base64
  }
};
```

### Detect Only

```javascript
const detectDocument = async (imageBase64) => {
  const response = await fetch('http://YOUR_SERVER:5000/api/detect', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: imageBase64 })
  });
  
  const result = await response.json();
  // result.corners = [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
};
```

## Response Format

### /api/scan Response

```json
{
  "success": true,
  "scan": "data:image/jpeg;base64,/9j/4AAQ...",
  "corners": [[100, 80], [540, 80], [540, 400], [100, 400]],
  "confidence": 0.95,
  "processing_time_ms": 45.2
}
```

## Environment Variables

- `MODEL_PATH` - Path to trained model weights
- `DEVICE` - Device to use (cuda/cpu/auto)

## Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt flask flask-cors

EXPOSE 5000
CMD ["python", "api/app.py", "--host", "0.0.0.0"]
```
