"""
PDF Export Utilities

Provides functions for exporting scanned documents to PDF
with optional OCR text layer for searchable documents.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union
from datetime import datetime

# Try to import PDF libraries
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from io import BytesIO
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Try to import OCR
try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


def export_to_pdf(
    images: Union[np.ndarray, List[np.ndarray]],
    output_path: str,
    page_size: str = 'A4',
    dpi: int = 300,
    add_ocr_layer: bool = False,
    title: Optional[str] = None
) -> bool:
    """
    Export scanned images to PDF.
    
    Args:
        images: Single image or list of images to include
        output_path: Path for output PDF
        page_size: 'A4' or 'letter'
        dpi: Resolution for images
        add_ocr_layer: Whether to add searchable text layer
        title: Optional PDF title metadata
        
    Returns:
        True if successful, False otherwise
    """
    if not REPORTLAB_AVAILABLE:
        print("Warning: reportlab not installed. Cannot export to PDF.")
        return _export_pdf_fallback(images, output_path)
    
    # Ensure images is a list
    if isinstance(images, np.ndarray):
        images = [images]
    
    # Get page size
    if page_size.upper() == 'A4':
        size = A4
    else:
        size = letter
    
    # Create PDF
    c = canvas.Canvas(output_path, pagesize=size)
    
    if title:
        c.setTitle(title)
    
    c.setAuthor("Document Scanner")
    c.setCreator("Shadow-Robust Document Scanner")
    
    page_width, page_height = size
    
    for i, img in enumerate(images):
        if i > 0:
            c.showPage()
        
        # Convert BGR to RGB
        if len(img.shape) == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Convert to PIL Image
        pil_img = Image.fromarray(img_rgb)
        
        # Calculate scaling to fit page
        img_width, img_height = pil_img.size
        scale_w = (page_width - 40) / img_width  # 20pt margin on each side
        scale_h = (page_height - 40) / img_height
        scale = min(scale_w, scale_h)
        
        new_width = img_width * scale
        new_height = img_height * scale
        
        # Center on page
        x = (page_width - new_width) / 2
        y = (page_height - new_height) / 2
        
        # Add image
        img_buffer = BytesIO()
        pil_img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        c.drawImage(
            ImageReader(img_buffer),
            x, y, new_width, new_height
        )
        
        # Add OCR layer if requested
        if add_ocr_layer and TESSERACT_AVAILABLE:
            _add_ocr_layer(c, img_rgb, x, y, new_width, new_height, scale)
    
    c.save()
    return True


def _add_ocr_layer(
    canvas_obj,
    image: np.ndarray,
    x: float,
    y: float,
    width: float,
    height: float,
    scale: float
) -> None:
    """
    Add invisible OCR text layer to PDF for searchability.
    """
    try:
        # Get OCR data with bounding boxes
        pil_img = Image.fromarray(image)
        data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
        
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            if not text:
                continue
            
            # Get coordinates
            text_x = data['left'][i] * scale + x
            text_y = height - (data['top'][i] + data['height'][i]) * scale + y
            text_height = data['height'][i] * scale
            
            # Set invisible text
            canvas_obj.setFillColorRGB(1, 1, 1, alpha=0)  # Invisible
            canvas_obj.setFont("Helvetica", max(6, text_height * 0.8))
            canvas_obj.drawString(text_x, text_y, text)
    
    except Exception as e:
        print(f"Warning: OCR layer failed: {e}")


def _export_pdf_fallback(
    images: Union[np.ndarray, List[np.ndarray]],
    output_path: str
) -> bool:
    """
    Fallback PDF export using OpenCV image saving.
    
    Creates individual image files instead of PDF.
    """
    output_dir = Path(output_path).parent
    base_name = Path(output_path).stem
    
    if isinstance(images, np.ndarray):
        images = [images]
    
    for i, img in enumerate(images):
        img_path = output_dir / f"{base_name}_page_{i+1}.png"
        cv2.imwrite(str(img_path), img)
    
    print(f"PDF export unavailable. Saved {len(images)} images instead.")
    return True


def extract_text_from_scan(
    image: np.ndarray,
    language: str = 'eng',
    preprocess: bool = True
) -> str:
    """
    Extract text from scanned document using OCR.
    
    Args:
        image: Scanned document image (BGR or grayscale)
        language: Tesseract language code
        preprocess: Whether to preprocess image for better OCR
        
    Returns:
        Extracted text
    """
    if not TESSERACT_AVAILABLE:
        return "OCR not available (pytesseract not installed)"
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Preprocess for better OCR
    if preprocess:
        # Binarization
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        gray = cv2.medianBlur(gray, 3)
    
    # Convert to PIL
    pil_img = Image.fromarray(gray)
    
    # OCR
    text = pytesseract.image_to_string(pil_img, lang=language)
    
    return text


def create_searchable_pdf(
    image: np.ndarray,
    output_path: str,
    language: str = 'eng'
) -> Tuple[bool, str]:
    """
    Create PDF with invisible text layer for searchability.
    
    Args:
        image: Scanned document image
        output_path: Output PDF path
        language: OCR language
        
    Returns:
        Tuple of (success, extracted_text)
    """
    # Extract text
    text = extract_text_from_scan(image, language)
    
    # Create PDF
    success = export_to_pdf([image], output_path, add_ocr_layer=True)
    
    return success, text


def batch_export_pdf(
    images: List[np.ndarray],
    output_path: str,
    pages_per_pdf: Optional[int] = None,
    add_page_numbers: bool = True
) -> List[str]:
    """
    Export multiple images to PDF(s).
    
    Args:
        images: List of scanned images
        output_path: Base output path
        pages_per_pdf: Max pages per PDF (None = all in one)
        add_page_numbers: Whether to add page numbers
        
    Returns:
        List of created PDF paths
    """
    if pages_per_pdf is None:
        pages_per_pdf = len(images)
    
    output_paths = []
    base_path = Path(output_path)
    
    for i in range(0, len(images), pages_per_pdf):
        batch = images[i:i + pages_per_pdf]
        
        if len(images) > pages_per_pdf:
            pdf_path = base_path.parent / f"{base_path.stem}_part{i//pages_per_pdf + 1}.pdf"
        else:
            pdf_path = base_path
        
        # Add page numbers if requested
        if add_page_numbers:
            batch = [_add_page_number(img, j + i + 1, len(images)) 
                    for j, img in enumerate(batch)]
        
        export_to_pdf(batch, str(pdf_path))
        output_paths.append(str(pdf_path))
    
    return output_paths


def _add_page_number(
    image: np.ndarray,
    page_num: int,
    total_pages: int
) -> np.ndarray:
    """Add page number to bottom of image."""
    result = image.copy()
    h, w = result.shape[:2]
    
    text = f"Page {page_num} of {total_pages}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    # Get text size
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Position at bottom center
    x = (w - text_w) // 2
    y = h - 15
    
    # Draw background rectangle
    cv2.rectangle(result, (x - 5, y - text_h - 5), (x + text_w + 5, y + 5), 
                 (255, 255, 255), -1)
    
    # Draw text
    cv2.putText(result, text, (x, y), font, font_scale, (0, 0, 0), thickness)
    
    return result


if __name__ == "__main__":
    print("Testing PDF export utilities...")
    
    # Create test image
    image = np.ones((800, 600, 3), dtype=np.uint8) * 255
    cv2.putText(image, "Test Document", (100, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv2.putText(image, "This is a test page", (100, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
    
    # Test text extraction
    if TESSERACT_AVAILABLE:
        text = extract_text_from_scan(image)
        print(f"Extracted text: {text[:50]}...")
    else:
        print("Tesseract not available, skipping OCR test")
    
    # Test PDF export
    if REPORTLAB_AVAILABLE:
        success = export_to_pdf(image, "test_output.pdf", title="Test Document")
        print(f"PDF export: {'Success' if success else 'Failed'}")
    else:
        print("ReportLab not available, PDF export will use fallback")
    
    print("Export tests complete!")
