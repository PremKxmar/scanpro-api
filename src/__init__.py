"""
Shadow-Robust Document Scanner

A mobile-first document scanner combining classical geometric vision 
with modern edge-AI for robust, ethical, offline document digitization.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .pipeline.scanner import DocumentScanner

__all__ = ["DocumentScanner"]
