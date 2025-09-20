import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_sauvola

def display_image(image, title=''):
    """Display an image for debugging purposes"""
    plt.figure(figsize=(10, 5))
    if len(image.shape) == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"debug_{title.replace(' ', '_')}.png")
    plt.close()

# ===== HISTOGRAM EQUALIZATION FUNCTIONS =====
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
# ===== END OF HISTOGRAM EQUALIZATION FUNCTIONS =====

def preprocess_image(image_path, enable_histogram_eq=False):
    """
    Preprocess the input image for better text extraction
    
    Args:
        image_path: Path to input image
        enable_histogram_eq: Whether to apply histogram equalization
        
    Returns:
        Tuple of (original image, preprocessed binary image)
    """
    # Read image
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    display_image(gray, 'Grayscale')
    
    # Apply histogram equalization if enabled
    if enable_histogram_eq:
        gray = apply_histogram_equalization(gray)
        display_image(gray, 'Histogram_Equalized')
    
    # Apply bilateral filter to preserve edges while reducing noise
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    display_image(filtered, 'Filtered')
    
    # Apply Sauvola thresholding (adaptive to local regions)
    window_size = 25
    thresh_sauvola = threshold_sauvola(filtered, window_size=window_size)
    binary = (filtered > thresh_sauvola).astype('uint8') * 255
    display_image(binary, 'Thresholded')
    
    # Invert image (text should be white on black)
    binary_inv = cv2.bitwise_not(binary)
    display_image(binary_inv, 'Inverted Binary')
    
    # Remove small noise
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel)
    display_image(cleaned, 'Cleaned')
    
    return original, cleaned

def segment_lines(binary_image):
    """Segment the image into lines of text"""
    # Get horizontal projection profile
    h_proj = np.sum(binary_image, axis=1)
    
    # Visualize projection profile
    plt.figure(figsize=(10, 4))
    plt.plot(h_proj, np.arange(binary_image.shape[0]))
    plt.gca().invert_yaxis()
    plt.title('Horizontal Projection Profile')
    plt.savefig('horizontal_projection.png')
    plt.close()
    
    # Find line boundaries using the projection profile
    thresh = np.max(h_proj) * 0.05  # Threshold for detecting line breaks
    line_boundaries = []
    in_line = False
    start = 0
    
    for i, count in enumerate(h_proj):
        if not in_line and count > thresh:
            # Line start
            in_line = True
            start = i
        elif in_line and count <= thresh:
            # Line end
            in_line = False
            # Only add if the line has a reasonable height
            if i - start >= 10:  # Minimum line height
                line_boundaries.append((start, i))
    
    # Handle the case where the last line extends to the end of the image
    if in_line:
        line_boundaries.append((start, len(h_proj) - 1))
    
    # Extract line images
    line_images = []
    debug_image = cv2.cvtColor(binary_image.copy(), cv2.COLOR_GRAY2BGR)
    
    for i, (y_start, y_end) in enumerate(line_boundaries):
        # Draw line boundary on debug image
        cv2.line(debug_image, (0, y_start), (binary_image.shape[1], y_start), (0, 255, 0), 2)
        cv2.line(debug_image, (0, y_end), (binary_image.shape[1], y_end), (0, 0, 255), 2)
        
        # Extract the line image
        line_img = binary_image[y_start:y_end, :]
        line_images.append((line_img, (y_start, y_end)))
    
    display_image(debug_image, 'Line Segmentation')
    return line_images

def segment_words(line_image, original_y_coords):
    """Segment a line image into words"""
    # Rest of the function remains unchanged
    y_start, y_end = original_y_coords
    
    # Get vertical projection profile
    v_proj = np.sum(line_image, axis=0)
    
    # Visualize projection profile
    plt.figure(figsize=(10, 2))
    plt.plot(v_proj)
    plt.title(f'Vertical Projection for Line {y_start}-{y_end}')
    plt.savefig(f'vertical_projection_line_{y_start}_{y_end}.png')
    plt.close()
    
    # Find word boundaries using the projection profile
    thresh = np.max(v_proj) * 0.1  # Threshold for detecting word breaks
    word_boundaries = []
    in_word = False
    start = 0
    
    for i, count in enumerate(v_proj):
        if not in_word and count > thresh:
            # Word start
            in_word = True
            start = i
        elif in_word and count <= thresh:
            # Word end
            in_word = False
            # Only add if the word has a reasonable width
            if i - start >= 5:  # Minimum word width
                word_boundaries.append((start, i))
    
    # Handle the case where the last word extends to the end of the line
    if in_word:
        word_boundaries.append((start, len(v_proj) - 1))
    
    # Extract word images
    word_images = []
    debug_image = cv2.cvtColor(line_image.copy(), cv2.COLOR_GRAY2BGR)
    
    for i, (x_start, x_end) in enumerate(word_boundaries):
        # Draw word boundary on debug image
        cv2.line(debug_image, (x_start, 0), (x_start, line_image.shape[0]), (0, 255, 0), 2)
        cv2.line(debug_image, (x_end, 0), (x_end, line_image.shape[0]), (0, 0, 255), 2)
        
        # Extract the word image with padding for better character segmentation
        padding = 2
        safe_x_start = max(0, x_start - padding)
        safe_x_end = min(line_image.shape[1], x_end + padding)
        word_img = line_image[:, safe_x_start:safe_x_end]
        
        # Store word with its coordinates in the original image
        global_coords = (
            (safe_x_start, safe_x_end),  # x coordinates
            (y_start, y_end)              # y coordinates
        )
        word_images.append((word_img, global_coords))
    
    display_image(debug_image, f'Word Segmentation Line {y_start}-{y_end}')
    return word_images

def segment_characters(word_image, original_coords):
    """Segment a word image into individual characters"""
    # Function remains unchanged
    (x_start, x_end), (y_start, y_end) = original_coords
    
    # Find contours in the word image
    contours, _ = cv2.findContours(word_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out noise contours
    valid_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # Filter based on area and aspect ratio
        if area > 20 and 0.1 < w/h < 10:
            valid_contours.append(contour)
    
    # Sort contours from left to right
    char_bounding_boxes = [cv2.boundingRect(c) for c in valid_contours]
    sorted_indices = sorted(range(len(char_bounding_boxes)), key=lambda i: char_bounding_boxes[i][0])
    sorted_contours = [valid_contours[i] for i in sorted_indices]
    sorted_bboxes = [char_bounding_boxes[i] for i in sorted_indices]
    
    # Create character images
    char_images = []
    debug_image = cv2.cvtColor(word_image.copy(), cv2.COLOR_GRAY2BGR)
    
    for i, (x, y, w, h) in enumerate(sorted_bboxes):
        # Draw rectangle around character
        cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0, 255, 0), 1)
        cv2.putText(debug_image, str(i+1), (x, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        
        # Get character ROI
        char_roi = word_image[y:y+h, x:x+w]
        
        # Calculate global coordinates
        global_x = x + x_start
        global_y = y + y_start
        global_coords = (global_x, global_y, w, h)
        
        # Store character image and its position
        char_images.append((char_roi, global_coords))
    
    word_id = f"{x_start}_{y_start}"
    display_image(debug_image, f'Character Segmentation Word {word_id}')
    return char_images

def normalize_characters(char_images):
    """Normalize character images for CNN input"""
    # Function remains unchanged
    normalized_chars = []
    
    for char_img, coords in char_images:
        # Add padding
        padding = 4
        h, w = char_img.shape
        padded = np.zeros((h + 2*padding, w + 2*padding), dtype=np.uint8)
        padded[padding:padding+h, padding:padding+w] = char_img
        
        # Find the best square crop
        h, w = padded.shape
        size = max(h, w)
        square = np.zeros((size, size), dtype=np.uint8)
        
        # Center the character in the square
        y_offset = (size - h) // 2
        x_offset = (size - w) // 2
        square[y_offset:y_offset+h, x_offset:x_offset+w] = padded
        
        # EMNIST requires 28x28 input
        resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Match EMNIST format (might need rotation/flip based on dataset)
        # Match EMNIST format: transpose, then flip vertically
        normalized = cv2.rotate(resized, cv2.ROTATE_90_CLOCKWISE)
        normalized = cv2.flip(normalized, 0)

        
        normalized_chars.append((normalized, coords))
    
    return normalized_chars

def extract_characters_from_image(image_path, enable_histogram_eq=False):
    """
    Extract individual characters from a handwritten text image
    
    Args:
        image_path: Path to input image
        enable_histogram_eq: Whether to apply histogram equalization
        
    Returns:
        Tuple of (original image, normalized characters)
    """
    # Preprocess the image
    original, binary = preprocess_image(image_path, enable_histogram_eq)
    
    # Segment into lines
    line_images = segment_lines(binary)
    
    # Segment each line into words
    all_word_images = []
    for line_img, line_coords in line_images:
        word_images = segment_words(line_img, line_coords)
        all_word_images.extend(word_images)
    
    # Segment each word into characters
    all_char_images = []
    for word_img, word_coords in all_word_images:
        char_images = segment_characters(word_img, word_coords)
        all_char_images.extend(char_images)
    
    # Normalize characters for the CNN
    normalized_chars = normalize_characters(all_char_images)
    
    # Visualize all normalized characters
    grid_size = int(np.ceil(np.sqrt(len(normalized_chars))))
    plt.figure(figsize=(15, 15))
    
    for i, (char_img, _) in enumerate(normalized_chars[:min(100, len(normalized_chars))]):
        if i < grid_size * grid_size:
            plt.subplot(grid_size, grid_size, i+1)
            plt.imshow(char_img, cmap='gray')
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('all_normalized_characters.png')
    plt.close()
    
    return original, normalized_chars