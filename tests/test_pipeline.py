"""
Unit tests for document scanning pipeline.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.scanner import DocumentScanner
from src.pipeline.detector import detect_document_classical, detect_document_edges
from src.pipeline.warper import warp_document, order_corners


class TestDocumentScanner:
    """Tests for main DocumentScanner class."""
    
    @pytest.fixture
    def scanner(self):
        """Create scanner instance."""
        return DocumentScanner(model_path=None, device='cpu')
    
    @pytest.fixture
    def test_image(self):
        """Create test image with document-like rectangle."""
        image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        # Draw a document rectangle
        pts = np.array([[100, 80], [540, 80], [540, 400], [100, 400]], dtype=np.int32)
        cv2.fillPoly(image, [pts], (220, 220, 220))
        return image
    
    def test_scanner_initialization(self, scanner):
        """Test scanner can be initialized."""
        assert scanner is not None
        assert scanner.device is not None
    
    def test_scan_returns_dict(self, scanner, test_image):
        """Test scan returns dictionary with expected keys."""
        result = scanner.scan(test_image)
        
        assert isinstance(result, dict)
        assert 'scan' in result
        assert 'corners' in result
        assert 'confidence' in result
        assert 'mask' in result


class TestClassicalDetector:
    """Tests for classical CV detector."""
    
    @pytest.fixture
    def document_image(self):
        """Create image with clear document."""
        image = np.ones((400, 600, 3), dtype=np.uint8) * 100
        pts = np.array([[80, 60], [520, 60], [520, 340], [80, 340]], dtype=np.int32)
        cv2.fillPoly(image, [pts], (220, 220, 220))
        return image
    
    def test_edge_detection_canny(self, document_image):
        """Test Canny edge detection."""
        edges = detect_document_edges(document_image, method='canny')
        
        assert edges.shape == document_image.shape[:2]
        assert edges.dtype == np.uint8
    
    def test_edge_detection_sobel(self, document_image):
        """Test Sobel edge detection."""
        edges = detect_document_edges(document_image, method='sobel')
        
        assert edges.shape == document_image.shape[:2]
    
    def test_classical_detection(self, document_image):
        """Test classical document detection."""
        corners, confidence = detect_document_classical(document_image)
        
        # Should detect something
        assert corners is not None or confidence == 0
        
        if corners is not None:
            assert corners.shape == (4, 2)
            assert 0 <= confidence <= 1


class TestWarper:
    """Tests for perspective warping."""
    
    def test_order_corners(self):
        """Test corner ordering."""
        # Unordered corners
        unordered = np.array([
            [100, 400],  # bottom-left (should be index 3)
            [500, 100],  # top-right (should be index 1)
            [100, 100],  # top-left (should be index 0)
            [500, 400],  # bottom-right (should be index 2)
        ], dtype=np.float32)
        
        ordered = order_corners(unordered)
        
        # Check ordering: TL, TR, BR, BL
        assert ordered[0][0] < ordered[1][0]  # TL.x < TR.x
        assert ordered[0][1] < ordered[3][1]  # TL.y < BL.y
        assert ordered[2][0] > ordered[3][0]  # BR.x > BL.x
    
    def test_warp_document(self):
        """Test document warping."""
        # Create test image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Define corners
        corners = np.array([
            [100, 80], [540, 80], [540, 400], [100, 400]
        ], dtype=np.float32)
        
        # Warp
        warped = warp_document(image, corners, output_size=(400, 500))
        
        assert warped.shape == (500, 400, 3)
    
    def test_warp_preserves_channels(self):
        """Test warping preserves color channels."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[:, :, 0] = 255  # Red channel
        
        corners = np.array([
            [0, 0], [639, 0], [639, 479], [0, 479]
        ], dtype=np.float32)
        
        warped = warp_document(image, corners, output_size=(320, 240))
        
        # Should still be predominantly red
        assert warped[:, :, 0].mean() > warped[:, :, 1].mean()


class TestIntegration:
    """Integration tests for full pipeline."""
    
    def test_full_pipeline(self):
        """Test complete scan pipeline."""
        # Create realistic test image
        image = np.ones((600, 800, 3), dtype=np.uint8) * 120
        
        # Add document
        doc_pts = np.array([[150, 100], [650, 120], [640, 480], [140, 460]], dtype=np.int32)
        cv2.fillPoly(image, [doc_pts], (230, 230, 230))
        
        # Add some text-like features
        for y in range(150, 450, 30):
            cv2.line(image, (180, y), (600, y), (50, 50, 50), 2)
        
        # Run scanner
        scanner = DocumentScanner(model_path=None, device='cpu')
        result = scanner.scan(image)
        
        # Should produce some output
        assert result is not None
        assert 'scan' in result
        assert 'confidence' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
