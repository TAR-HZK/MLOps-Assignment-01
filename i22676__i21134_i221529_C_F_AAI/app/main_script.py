import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import pyttsx3
from difflib import get_close_matches
import re
import nltk

# Import PaddleOCR integration
try:
    from paddle_ocr import process_image as paddle_process_image
    from paddle_ocr import text_to_speech
    print("Successfully imported PaddleOCR integration")
except ImportError:
    print("Warning: paddle_ocr module not found. Ensure it is installed or available.")
    paddle_process_image = None
    text_to_speech = None

# Define a dummy EMNISTModel class to handle import statements in old code
# that might still be referencing it
class EMNISTModel:
    def __init__(self, *args, **kwargs):
        print("Using dummy EMNISTModel - actual processing will use PaddleOCR")
    
    def load_state_dict(self, *args, **kwargs):
        pass
    
    def eval(self):
        pass
    
    def to(self, *args, **kwargs):
        return self

# Try to import original components for backward compatibility
try:
    from preprocessing import extract_characters_from_image
    HAS_ORIGINAL_MODEL = True
    print("Successfully imported original preprocessing module")
except ImportError:
    print("Warning: Original preprocessing module not found")
    HAS_ORIGINAL_MODEL = False

def process_image(image_path, model_path=None, enable_spell_correction=False, enable_histogram_eq=False):
    """
    Process an image through the OCR pipeline
    
    This is a wrapper function that uses PaddleOCR integration
    but maintains compatibility with the original function signature
    
    Args:
        image_path: Path to the input image
        model_path: Path to model (kept for compatibility)
        enable_spell_correction: Whether to enable spelling correction
        enable_histogram_eq: Whether to apply histogram equalization
        
    Returns:
        Recognized text as a string
    """
    # Check if image exists
    if not os.path.exists(image_path):
        return f"Image file not found: {image_path}"
    
    # Log which model is being used
    if model_path and HAS_ORIGINAL_MODEL:
        print(f"Using original EMNIST model: {model_path}")
        # Original model processing could be implemented here if needed
        # For now, we'll always use PaddleOCR
        print("Switching to PaddleOCR for better accuracy...")
    
    # Check if PaddleOCR integration is available
    if paddle_process_image is None:
        return "PaddleOCR integration is not available. Please check paddle_ocr.py"
    
    # Use PaddleOCR for processing with the new histogram equalization option
    return paddle_process_image(image_path, model_path, enable_spell_correction, enable_histogram_eq)

# If text_to_speech is not imported from paddle_ocr, define it here
if text_to_speech is None:
    def text_to_speech(text):
        """Convert text to speech using pyttsx3"""
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            return True
        except Exception as e:
            print(f"Text-to-speech error: {str(e)}")
            return False