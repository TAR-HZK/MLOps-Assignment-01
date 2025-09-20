import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR
import nltk
from difflib import get_close_matches
import re
import pyttsx3

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('words')
    nltk.download('brown')



class PaddleOCRModel:
    def __init__(self, use_gpu=False):
        """Initialize the PaddleOCR model"""
        self.use_gpu = use_gpu
        self.ocr = None
        self.initialize_model()
        
    def initialize_model(self):
        """Initialize the PaddleOCR model with optimizations for handwriting"""
        try:
            # Initialize with optimizations for handwriting
            self.ocr = PaddleOCR(
                use_angle_cls=True,        # Detect text orientation
                lang='en',                 # English language
                use_gpu=self.use_gpu,      # GPU usage based on availability
                show_log=False,            # Don't show detailed logs
                # Optimization parameters for handwriting
                det_db_thresh=0.3,         # Lower detection threshold for handwriting
                det_db_box_thresh=0.5,     # Box threshold
                det_limit_side_len=960,    # Limit image size for faster processing
                max_batch_size=10,         # Batch size for processing
                det_limit_type='max'       # Limit by max size
            )
            print("PaddleOCR model initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing PaddleOCR model: {str(e)}")
            return False
    
    def process_image(self, image_path):
        """Process an image with PaddleOCR and return detection results"""
        if self.ocr is None:
            print("Model not initialized")
            return None
        
        try:
            # Read image
            if isinstance(image_path, str):
                if not os.path.exists(image_path):
                    print(f"Image file not found: {image_path}")
                    return None
                image = cv2.imread(image_path)
            else:
                # Assume image_path is already a numpy array
                image = image_path
                
            # Run OCR
            result = self.ocr.ocr(image, cls=True)
            
            # If no results, return empty
            if not result or len(result) == 0:
                print("No text detected in the image")
                return []
                
            return result
        except Exception as e:
            print(f"Error processing image with PaddleOCR: {str(e)}")
            return None

def extract_text_and_coordinates(ocr_result):
    """Extract text and bounding box coordinates from OCR results"""
    text_items = []
    
    # Handle both list of lists structure (for older PaddleOCR versions)
    # and direct list structure (for newer versions)
    if ocr_result and isinstance(ocr_result, list):
        # PaddleOCR can return results in different formats based on version
        if len(ocr_result) > 0 and isinstance(ocr_result[0], list):
            # Format: [[[points], (text, confidence)], ...]
            for line in ocr_result:
                for item in line:
                    if len(item) == 2 and isinstance(item[0], list) and isinstance(item[1], tuple):
                        bbox = item[0]
                        text, confidence = item[1]
                        
                        # Calculate bounding box in (x, y, w, h) format
                        x_values = [point[0] for point in bbox]
                        y_values = [point[1] for point in bbox]
                        x = min(x_values)
                        y = min(y_values)
                        w = max(x_values) - x
                        h = max(y_values) - y
                        
                        text_items.append({
                            'text': text,
                            'confidence': confidence,
                            'bbox': (int(x), int(y), int(w), int(h)),
                            'raw_bbox': bbox
                        })
        else:
            # Format might be different in newer versions
            for item in ocr_result:
                if len(item) >= 2:
                    bbox = item[0]
                    text, confidence = item[1]
                    
                    # Calculate bounding box in (x, y, w, h) format
                    x_values = [point[0] for point in bbox]
                    y_values = [point[1] for point in bbox]
                    x = min(x_values)
                    y = min(y_values)
                    w = max(x_values) - x
                    h = max(y_values) - y
                    
                    text_items.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': (int(x), int(y), int(w), int(h)),
                        'raw_bbox': bbox
                    })
    
    return text_items

def spelling_correction(words, enable_correction=False, confidence_threshold=0.8):
    """Apply spelling correction to recognized words, with option to bypass"""
    # If correction is disabled, return the original words
    if not enable_correction:
        return words
        
    # Load dictionary of words
    try:
        nltk_words = set(nltk.corpus.words.words() + nltk.corpus.brown.words())
        dictionary = set(word.lower() for word in nltk_words if len(word) > 1)
    except:
        print("Warning: NLTK word corpus not available. Spell check limited.")
        # Fallback to basic dictionary
        dictionary = set(['the', 'a', 'an', 'is', 'are', 'was', 'were', 'hello', 'world', 
                          'this', 'that', 'test', 'handwriting', 'voice', 'project', 'demo',
                          'to', 'from', 'with', 'by', 'and', 'or', 'for', 'in', 'on', 'at'])
    
    corrected_words = []
    
    for word in words:
        # Skip correction for very short words, numbers, and punctuation
        if len(word) <= 1 or word.isdigit() or re.match(r'^[.,!?:;"\'\-]+$', word):
            corrected_words.append(word)
            continue
        
        # Check if the word is already in the dictionary
        if word.lower() in dictionary:
            corrected_words.append(word)
            continue
        
        # Try to find close matches
        close_matches = get_close_matches(word.lower(), dictionary, n=1, cutoff=confidence_threshold)
        
        if close_matches:
            # Compare the match with the original word
            match = close_matches[0]
            # Only replace if it's significantly better (using a higher threshold)
            similarity_ratio = len(set(word.lower()) & set(match)) / max(len(set(word.lower())), len(set(match)))
            
            if similarity_ratio > 0.7:  # High similarity required to make the correction
                # Preserve capitalization
                if word[0].isupper():
                    corrected_words.append(match.capitalize())
                else:
                    corrected_words.append(match)
            else:
                # Not similar enough, keep the original
                corrected_words.append(word)
        else:
            # No close match found, keep the original word
            corrected_words.append(word)
    
    return corrected_words

# ===== HISTOGRAM EQUALIZATION CODE INTEGRATION =====
def calculate_histogram(image):
    """
    Calculate histogram of grayscale image
    
    Args:
        image: Grayscale image (2D numpy array)
    
    Returns:
        List of 256 values representing pixel counts
    """
    rows, cols = np.shape(image)
    histogram = [0] * 256
    for i in range(rows):
        for j in range(cols):
            pixel_value = image[i, j]
            histogram[pixel_value] += 1
    return histogram

def calculate_pdf(histogram, image):
    """
    Calculate probability density function from histogram
    
    Args:
        histogram: Image histogram
        image: Grayscale image
    
    Returns:
        List of 256 probability values
    """
    rows, cols = np.shape(image)
    pdf = [0] * 256
    for i in range(256):
        count = histogram[i]
        p = count / (rows * cols)
        pdf[i] = p
    return pdf

def calculate_cdf(pdf):
    """
    Calculate cumulative distribution function from PDF
    
    Args:
        pdf: Probability density function
    
    Returns:
        Numpy array with CDF values
    """
    cdf = np.zeros(len(pdf))
    cdf[0] = pdf[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + pdf[i]
    return cdf

def apply_histogram_equalization(image):
    """
    Apply histogram equalization to enhance image contrast
    
    Args:
        image: Grayscale image (2D numpy array)
    
    Returns:
        Equalized image
    """
    # Calculate histogram, PDF, and CDF
    histogram = calculate_histogram(image)
    pdf = calculate_pdf(histogram, image)
    cdf = calculate_cdf(pdf)
    
    # Apply equalization transformation
    rows, cols = np.shape(image)
    transformation_func = np.around(cdf * 255)
    equalized_image = np.copy(image)
    
    # Apply transformation to each pixel
    for i in range(rows):
        for j in range(cols):
            pixel_value = image[i, j]
            equalized_image[i, j] = transformation_func[pixel_value].astype(np.uint8)
    
    return equalized_image

def visualize_histogram_equalization(original, equalized):
    """
    Visualize the original and equalized images with their histograms
    
    Args:
        original: Original grayscale image
        equalized: Equalized grayscale image
    
    Returns:
        Visualization figure that can be saved or displayed
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Original histogram
    axes[0, 1].hist(original.ravel(), 256, [0, 256], color='b', alpha=0.7)
    axes[0, 1].set_title('Original Histogram')
    axes[0, 1].set_xlim([0, 256])
    
    # Equalized image
    axes[1, 0].imshow(equalized, cmap='gray')
    axes[1, 0].set_title('Equalized Image')
    axes[1, 0].axis('off')
    
    # Equalized histogram
    axes[1, 1].hist(equalized.ravel(), 256, [0, 256], color='r', alpha=0.7)
    axes[1, 1].set_title('Equalized Histogram')
    axes[1, 1].set_xlim([0, 256])
    
    plt.tight_layout()
    plt.savefig('histogram_equalization_visualization.png')
    
    return fig
# ===== END OF HISTOGRAM EQUALIZATION CODE =====

def apply_language_model(words):
    """Apply basic language model rules to improve the recognized text"""
    corrected_text = []
    
    for i, word in enumerate(words):
        # Apply common word-specific corrections
        lower_word = word.lower()
        
        # Common OCR confusions
        if lower_word == 'l' and (i == 0 or words[i-1][-1] in '.!?'):
            corrected_text.append('I')  # Single 'l' at start of sentence is likely 'I'
        elif re.match(r'[0o]ne', lower_word):
            corrected_text.append('one')
        elif re.match(r'[0o]f', lower_word) and len(word) == 2:
            corrected_text.append('of')
        elif re.match(r't[0o]', lower_word) and len(word) == 2:
            corrected_text.append('to')
        elif re.match(r'th[l1]s', lower_word):
            corrected_text.append('this')
        else:
            corrected_text.append(word)
    
    # Join words and apply sentence-level corrections
    text = ' '.join(corrected_text)
    
    # Fix spacing after punctuation
    text = re.sub(r'(\w)([,.!?:;])(\w)', r'\1\2 \3', text)
    
    # Ensure first letter of sentences is capitalized
    text = re.sub(r'(^|[.!?]\s+)([a-z])', lambda p: p.group(1) + p.group(2).upper(), text)
    
    return text

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

def visualize_results(image_path, ocr_results, final_text):
    """Visualize recognition results"""
    # Read the original image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Failed to read image from {image_path}")
        return None
        
    # Create a copy for visualization
    result_image = original_image.copy()
    
    # Draw detected text regions
    for item in ocr_results:
        bbox = item['raw_bbox']
        text = item['text']
        confidence = item['confidence']
        
        # Convert points to numpy array for drawing
        pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
        
        # Color based on confidence: green for high, red for low
        color = (0, int(255 * confidence), int(255 * (1 - confidence)))
        
        # Draw polygon around text
        cv2.polylines(result_image, [pts], True, color, 2)
        
        # Add text label
        cv2.putText(result_image, text, (int(bbox[0][0]), int(bbox[0][1])-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Create composite visualization
    plt.figure(figsize=(15, 10))
    
    # Original image with recognition
    plt.subplot(2, 1, 1)
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title('PaddleOCR Recognition Results')
    plt.axis('off')
    
    # Recognition text
    plt.subplot(2, 1, 2)
    plt.text(0.5, 0.5, f"Recognized Text:\n{final_text}", 
             horizontalalignment='center', verticalalignment='center',
             fontsize=14, wrap=True)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('recognition_results.png')
    plt.show()
    
    return result_image

def process_image(image_path, model_path=None, enable_spell_correction=False, enable_histogram_eq=False):
    """Process an image through the entire OCR pipeline with PaddleOCR"""
    # model_path parameter is kept for compatibility with existing code, but not used
    
    # Initialize PaddleOCR model
    paddle_model = PaddleOCRModel(use_gpu=False)  # Set to True if you have GPU support
    
    # Process the image
    print("Processing image with PaddleOCR...")
    
    # Apply histogram equalization if enabled
    if enable_histogram_eq:
        original_image = cv2.imread(image_path)
        if original_image is None:
            return "Could not read image"
        
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        equalized_image = apply_histogram_equalization(gray_image)
        
        # Convert back to BGR for PaddleOCR
        equalized_bgr = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)
        
        # Save temporarily or use in-memory
        temp_path = "temp_equalized.jpg"
        cv2.imwrite(temp_path, equalized_bgr)
        ocr_result = paddle_model.process_image(temp_path)
        
        # Optionally remove the temporary file
        try:
            os.remove(temp_path)
        except:
            pass
    else:
        ocr_result = paddle_model.process_image(image_path)
    
    if not ocr_result or len(ocr_result) == 0:
        return "No text detected in the image"
    
    # Extract text and coordinates
    text_items = extract_text_and_coordinates(ocr_result)
    if not text_items:
        return "Failed to extract text from OCR results"
    
    # Sort text items from top to bottom, left to right (natural reading order)
    text_items.sort(key=lambda x: (x['bbox'][1], x['bbox'][0]))
    
    # Extract text only for further processing
    raw_words = [item['text'] for item in text_items]
    print(f"Raw detected words: {raw_words}")
    
    # Join small fragments that might be parts of same word
    # This is a simple approach - PaddleOCR usually does a good job with word segmentation
    words = []
    current_word = ""
    
    for word in raw_words:
        if len(word) <= 2 and not word.isdigit() and not word in ".,:;!?":
            current_word += word
        else:
            if current_word:
                words.append(current_word)
                current_word = ""
            words.append(word)
    
    if current_word:  # Add the last word if any
        words.append(current_word)
    
    print(f"Processed words: {words}")
    
    # Apply spelling correction only if enabled
    if enable_spell_correction:
        corrected_words = spelling_correction(words, enable_correction=True, confidence_threshold=0.8)
        print(f"Spell-corrected words: {corrected_words}")
    else:
        corrected_words = words
        print(f"Spell correction skipped")
    
    # Apply language model
    final_text = apply_language_model(corrected_words)
    print(f"Final text: {final_text}")
    
    # Visualize results
    result_image = visualize_results(image_path, text_items, final_text)
    
    return final_text

if __name__ == "__main__":
    # Test with a sample image
    image_path = "sample_images/hw2.jpg"
    if os.path.exists(image_path):
        # Set enable_spell_correction=False to skip spell checking
        text = process_image(image_path, enable_spell_correction=False)
        print("\nFinal recognized text:", text)
        
        # Test text-to-speech
        if text:
            text_to_speech(text)
    else:
        print(f"Test image not found at {image_path}")