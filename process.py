import cv2
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_pupil_response(image_bytes):
    try:
        # Decode image
        npimg = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if frame is None:
            logger.error("Failed to decode image")
            return create_empty_result()
        
        logger.info(f"Image decoded: {frame.shape}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Try multiple detection approaches
        result = detect_with_hough_circles(gray)
        if result["pupil_area"] > 0:
            logger.info("Pupil detected with Hough circles")
            return result
        
        result = detect_with_adaptive_threshold(gray)
        if result["pupil_area"] > 0:
            logger.info("Pupil detected with adaptive threshold")
            return result
        
        result = detect_with_basic_threshold(gray)
        if result["pupil_area"] > 0:
            logger.info("Pupil detected with basic threshold")
            return result
        
        logger.warning("No pupil detected with any method")
        return create_empty_result()
        
    except Exception as e:
        logger.error(f"Error in pupil analysis: {str(e)}")
        return create_empty_result()

def detect_with_hough_circles(gray):
    """Detect pupils using HoughCircles - most robust for circular pupils"""
    try:
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Detect circles
        circles = cv2.HoughCircles(
            enhanced,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=200
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            # Find the best circle (largest, most centered)
            best_circle = None
            best_score = 0
            
            for (x, y, r) in circles:
                # Score based on size and position
                area = np.pi * r * r
                center_score = 1.0 / (1.0 + np.sqrt((x - gray.shape[1]/2)**2 + (y - gray.shape[0]/2)**2) / 100)
                size_score = min(area / 1000, 1.0)  # Normalize area
                total_score = area * center_score * size_score
                
                if total_score > best_score and 50 < area < 10000:
                    best_circle = (x, y, r)
                    best_score = total_score
            
            if best_circle:
                x, y, r = best_circle
                area = np.pi * r * r
                return {
                    "pupil_area": round(area, 2),
                    "center": [round(float(x), 2), round(float(y), 2)],
                    "axes": [round(float(r*2), 2), round(float(r*2), 2)],
                    "angle": 0.0,
                    "circularity": 1.0  # Circles are perfectly circular
                }
    
    except Exception as e:
        logger.error(f"Error in Hough circles detection: {str(e)}")
    
    return create_empty_result()

def detect_with_adaptive_threshold(gray):
    """Detect pupils using adaptive thresholding"""
    try:
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return find_best_ellipse(contours)
    
    except Exception as e:
        logger.error(f"Error in adaptive threshold detection: {str(e)}")
    
    return create_empty_result()

def detect_with_basic_threshold(gray):
    """Detect pupils using basic thresholding with multiple thresholds"""
    try:
        # Try multiple threshold values
        threshold_values = [20, 30, 40, 50, 60]
        
        for thresh_val in threshold_values:
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply threshold
            _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            result = find_best_ellipse(contours)
            if result["pupil_area"] > 0:
                logger.info(f"Pupil found with threshold {thresh_val}")
                return result
    
    except Exception as e:
        logger.error(f"Error in basic threshold detection: {str(e)}")
    
    return create_empty_result()

def find_best_ellipse(contours):
    """Find the best ellipse from contours"""
    best_ellipse = None
    best_area = 0
    best_circularity = 0
    
    for cnt in contours:
        if len(cnt) >= 5:
            try:
                ellipse = cv2.fitEllipse(cnt)
                (x, y), (MA, ma), angle = ellipse
                
                # Calculate area and circularity
                area = np.pi * (MA / 2) * (ma / 2)
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                
                # More lenient criteria
                if (20 < area < 15000 and  # Wider area range
                    circularity > 0.3 and   # More lenient circularity
                    MA > 0 and ma > 0 and   # Valid axes
                    (MA/ma) < 3):           # Not too elongated
                    
                    if area > best_area:
                        best_ellipse = ellipse
                        best_area = area
                        best_circularity = circularity
            
            except Exception as e:
                continue  # Skip invalid contours
    
    if best_ellipse:
        (x, y), (MA, ma), angle = best_ellipse
        return {
            "pupil_area": round(best_area, 2),
            "center": [round(float(x), 2), round(float(y), 2)],
            "axes": [round(float(MA), 2), round(float(ma), 2)],
            "angle": round(float(angle), 2),
            "circularity": round(best_circularity, 3)
        }
    
    return create_empty_result()

def create_empty_result():
    """Create empty result when no pupil is detected"""
    return {
        "pupil_area": 0,
        "center": None,
        "axes": None,
        "angle": None,
        "circularity": 0
    }