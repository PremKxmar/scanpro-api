"""
Flask REST API for Document Scanner

This provides a REST API that can be consumed by mobile apps
(React Native, Flutter, or any HTTP client).

Usage:
    python api/app.py

Endpoints:
    POST /api/scan         - Scan a document image
    POST /api/detect       - Detect document corners only
    GET  /api/health       - Health check
    GET  /api/info         - API info and capabilities
"""

import os
import sys
import io
import base64
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from src.pipeline.scanner import DocumentScanner
from src.utils.export import export_to_pdf

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for mobile apps

# Initialize scanner (lazy load)
_scanner = None


def get_scanner() -> DocumentScanner:
    """Get or create scanner instance."""
    global _scanner
    if _scanner is None:
        model_path = os.environ.get('MODEL_PATH', None)
        device = os.environ.get('DEVICE', 'auto')
        _scanner = DocumentScanner(model_path=model_path, device=device)
    return _scanner


def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 string to OpenCV image."""
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Decode
    img_bytes = base64.b64decode(base64_string)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    return image


def encode_image_base64(image: np.ndarray, format: str = 'jpeg') -> str:
    """Encode OpenCV image to base64 string."""
    if format.lower() == 'jpeg':
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    else:
        _, buffer = cv2.imencode('.png', image)
    
    base64_string = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/{format};base64,{base64_string}"


# =============================================================================
# API Endpoints
# =============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'service': 'document-scanner-api'
    })


@app.route('/api/info', methods=['GET'])
def api_info():
    """Get API information and capabilities."""
    scanner = get_scanner()
    return jsonify({
        'name': 'Shadow-Robust Document Scanner API',
        'version': '1.0.0',
        'device': str(scanner.device),
        'capabilities': [
            'document_detection',
            'shadow_removal',
            'perspective_correction',
            'pdf_export'
        ],
        'endpoints': {
            '/api/scan': 'POST - Full document scanning',
            '/api/detect': 'POST - Detect document corners',
            '/api/enhance': 'POST - Enhance document image',
            '/api/health': 'GET - Health check',
            '/api/info': 'GET - API information'
        }
    })


@app.route('/api/scan', methods=['POST'])
def scan_document():
    """
    Scan a document from image.
    
    Request body (JSON):
    {
        "image": "base64_encoded_image_string",
        "options": {
            "remove_shadows": true,
            "enhance": true,
            "output_format": "jpeg"  // or "png"
        }
    }
    
    Response:
    {
        "success": true,
        "scan": "base64_encoded_scan",
        "corners": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
        "confidence": 0.95,
        "processing_time_ms": 45.2
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: image'
            }), 400
        
        # Decode image
        image = decode_base64_image(data['image'])
        if image is None:
            return jsonify({
                'success': False,
                'error': 'Failed to decode image'
            }), 400
        
        # Get options
        options = data.get('options', {})
        remove_shadows = options.get('remove_shadows', True)
        enhance = options.get('enhance', True)
        output_format = options.get('output_format', 'jpeg')
        
        # Process
        scanner = get_scanner()
        start_time = time.time()
        
        result = scanner.scan(
            image,
            remove_shadows=remove_shadows,
            enhance=enhance
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Prepare response
        response = {
            'success': True,
            'confidence': float(result['confidence']),
            'processing_time_ms': round(processing_time, 2)
        }
        
        if result['corners'] is not None:
            response['corners'] = result['corners'].tolist()
        else:
            response['corners'] = None
        
        if result['scan'] is not None:
            response['scan'] = encode_image_base64(result['scan'], output_format)
        else:
            response['scan'] = None
            response['message'] = 'Document not detected'
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/detect', methods=['POST'])
def detect_document():
    """
    Detect document corners without full scanning.
    
    Request body (JSON):
    {
        "image": "base64_encoded_image_string"
    }
    
    Response:
    {
        "success": true,
        "detected": true,
        "corners": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
        "confidence": 0.95,
        "mask": "base64_encoded_mask" (optional)
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: image'
            }), 400
        
        # Decode image
        image = decode_base64_image(data['image'])
        if image is None:
            return jsonify({
                'success': False,
                'error': 'Failed to decode image'
            }), 400
        
        # Process (scan but only return detection info)
        scanner = get_scanner()
        result = scanner.scan(image, remove_shadows=False, enhance=False)
        
        response = {
            'success': True,
            'detected': result['corners'] is not None,
            'confidence': float(result['confidence'])
        }
        
        if result['corners'] is not None:
            response['corners'] = result['corners'].tolist()
        
        if 'mask' in result and result['mask'] is not None:
            # Optionally include mask
            include_mask = data.get('include_mask', False)
            if include_mask:
                response['mask'] = encode_image_base64(result['mask'], 'png')
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/enhance', methods=['POST'])
def enhance_image():
    """
    Enhance a document image (shadow removal, etc).
    
    Request body (JSON):
    {
        "image": "base64_encoded_image_string",
        "options": {
            "remove_shadows": true,
            "sharpen": true,
            "denoise": true
        }
    }
    """
    try:
        from src.preprocessing.shadow_removal import enhance_document
        
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: image'
            }), 400
        
        # Decode image
        image = decode_base64_image(data['image'])
        if image is None:
            return jsonify({
                'success': False,
                'error': 'Failed to decode image'
            }), 400
        
        # Get options
        options = data.get('options', {})
        
        # Enhance
        enhanced = enhance_document(
            image,
            remove_shadows=options.get('remove_shadows', True),
            sharpen=options.get('sharpen', True),
            denoise=options.get('denoise', True)
        )
        
        return jsonify({
            'success': True,
            'enhanced': encode_image_base64(enhanced, 'jpeg')
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/export-pdf', methods=['POST'])
def export_pdf():
    """
    Export scanned image(s) to PDF.
    
    Request body (JSON):
    {
        "images": ["base64_image1", "base64_image2", ...],
        "options": {
            "page_size": "A4",  // or "letter"
            "add_ocr": false
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'images' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: images'
            }), 400
        
        # Decode images
        images = []
        for img_str in data['images']:
            img = decode_base64_image(img_str)
            if img is not None:
                images.append(img)
        
        if not images:
            return jsonify({
                'success': False,
                'error': 'No valid images provided'
            }), 400
        
        # Get options
        options = data.get('options', {})
        
        # Create PDF in memory
        output_buffer = io.BytesIO()
        temp_path = '/tmp/scan_output.pdf'
        
        export_to_pdf(
            images,
            temp_path,
            page_size=options.get('page_size', 'A4'),
            add_ocr_layer=options.get('add_ocr', False)
        )
        
        # Read and encode
        with open(temp_path, 'rb') as f:
            pdf_data = base64.b64encode(f.read()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'pdf': f"data:application/pdf;base64,{pdf_data}",
            'pages': len(images)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Document Scanner API")
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--model', type=str, default=None)
    
    args = parser.parse_args()
    
    if args.model:
        os.environ['MODEL_PATH'] = args.model
    
    print("=" * 60)
    print("  Document Scanner API")
    print("=" * 60)
    print(f"\nStarting server on http://{args.host}:{args.port}")
    print("\nEndpoints:")
    print("  POST /api/scan     - Full document scanning")
    print("  POST /api/detect   - Detect corners only")
    print("  POST /api/enhance  - Enhance image")
    print("  POST /api/export-pdf - Export to PDF")
    print("  GET  /api/health   - Health check")
    print("  GET  /api/info     - API info")
    print("-" * 60)
    
    app.run(host=args.host, port=args.port, debug=args.debug)
