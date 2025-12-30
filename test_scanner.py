"""
Quick test script for the document scanner.
"""
import cv2
import numpy as np
from src.pipeline.scanner import DocumentScanner

print('='*50)
print('  Document Scanner Demo')
print('='*50)

# Create test image with a document
print('\nCreating test image...')
image = np.ones((600, 800, 3), dtype=np.uint8) * 120

# Add document with perspective distortion
doc_pts = np.array([[120, 80], [680, 100], [660, 500], [100, 480]], dtype=np.int32)
cv2.fillPoly(image, [doc_pts], (230, 230, 230))

# Add text-like lines
for y in range(150, 450, 40):
    cv2.line(image, (160, y), (620, y), (50, 50, 50), 2)
cv2.putText(image, 'TEST DOCUMENT', (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

# Save input image
cv2.imwrite('test_input.jpg', image)
print('Saved: test_input.jpg')

# Initialize scanner
print('\nInitializing scanner (may take a moment to load models)...')
scanner = DocumentScanner(device='cpu')
print(f'Device: {scanner.device}')

# Run scan
print('\nScanning document...')
result = scanner.scan(image)

# Print results
print('\n' + '='*50)
print('  SCAN RESULTS')
print('='*50)
print(f'Scan shape: {result["scan"].shape}')
print(f'Confidence: {result["confidence"]:.4f}')
print(f'Corners detected: {result["corners"] is not None}')
if result['corners'] is not None:
    print('Corner positions:')
    for i, c in enumerate(result['corners']):
        print(f'  Corner {i}: ({c[0]:.1f}, {c[1]:.1f})')

# Save output
cv2.imwrite('test_output_scan.jpg', result['scan'])
print(f'\nSaved: test_output_scan.jpg')
cv2.imwrite('test_output_mask.jpg', result['mask'])
print('Saved: test_output_mask.jpg')

print('\n' + '='*50)
print('  Demo Complete!')
print('='*50)
