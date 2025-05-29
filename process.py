import cv2
import numpy as np
import logging
import tempfile
import os
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_pupil_video(video_bytes: bytes) -> Dict[str, Any]:
    """
    Analyze pupil response from entire video file
    Returns time-series data of pupil measurements
    """
    temp_video_path = None
    
    try:
        # Save video bytes to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mov') as temp_file:
            temp_file.write(video_bytes)
            temp_video_path = temp_file.name
        
        logger.info(f"Processing video file: {len(video_bytes)} bytes")
        
        # Open video with OpenCV
        cap = cv2.VideoCapture(temp_video_path)
        
        if not cap.isOpened():
            logger.error("Failed to open video file")
            return create_empty_video_result()
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        logger.info(f"Video: {frame_count} frames, {fps:.2f} FPS, {duration:.2f}s duration")
        
        # Process video frames
        pupil_data = []
        frame_number = 0
        
        # Sample every few frames for efficiency (e.g., every 3rd frame)
        sample_interval = max(1, int(fps / 10))  # ~10 samples per second
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process sampled frames
            if frame_number % sample_interval == 0:
                timestamp = frame_number / fps if fps > 0 else 0
                
                # Analyze this frame
                pupil_result = analyze_single_frame(frame)
                
                if pupil_result["pupil_area"] > 0:
                    pupil_data.append({
                        "frame_number": frame_number,
                        "timestamp": round(timestamp, 3),
                        "pupil_area": pupil_result["pupil_area"],
                        "center": pupil_result["center"],
                        "axes": pupil_result["axes"],
                        "angle": pupil_result["angle"],
                        "circularity": pupil_result["circularity"]
                    })
                    
                    logger.info(f"Frame {frame_number}: pupil_area = {pupil_result['pupil_area']:.1f}")
            
            frame_number += 1
        
        cap.release()
        
        # Calculate summary statistics
        if pupil_data:
            areas = [d["pupil_area"] for d in pupil_data]
            summary = {
                "average_area": round(np.mean(areas), 2),
                "min_area": round(np.min(areas), 2),
                "max_area": round(np.max(areas), 2),
                "std_area": round(np.std(areas), 2),
                "area_range": round(np.max(areas) - np.min(areas), 2),
                "detection_rate": round(len(pupil_data) / (frame_number / sample_interval) * 100, 1)
            }
        else:
            summary = {
                "average_area": 0,
                "min_area": 0,
                "max_area": 0,
                "std_area": 0,
                "area_range": 0,
                "detection_rate": 0
            }
        
        result = {
            "success": len(pupil_data) > 0,
            "video_info": {
                "duration": round(duration, 2),
                "fps": round(fps, 2),
                "total_frames": frame_count,
                "processed_frames": len(pupil_data)
            },
            "summary": summary,
            "pupil_data": pupil_data[:50]  # Limit to 50 data points for response size
        }
        
        logger.info(f"Analysis complete: {len(pupil_data)} pupil detections from {frame_number} frames")
        return result
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return create_empty_video_result()
        
    finally:
        # Clean up temporary file
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.unlink(temp_video_path)
            except:
                pass

def analyze_single_frame(frame: np.ndarray) -> Dict[str, Any]:
    """Analyze a single frame for pupil detection"""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Try multiple detection methods (same as before)
        result = detect_with_hough_circles(gray)
        if result["pupil_area"] > 0:
            return result
        
        result = detect_with_adaptive_threshold(gray)
        if result["pupil_area"] > 0:
            return result
        
        result = detect_with_basic_threshold(gray)
        if result["pupil_area"] > 0:
            return result
        
        return create_empty_result()
        
    except Exception as e:
        logger.error(f"Error analyzing frame: {str(e)}")
        return create_empty_result()

def detect_with_hough_circles(gray):
    """Detect pupils using HoughCircles"""
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
            
            # Find the best circle
            best_circle = None
            best_score = 0
            
            for (x, y, r) in circles:
                area = np.pi * r * r
                center_score = 1.0 / (1.0 + np.sqrt((x - gray.shape[1]/2)**2 + (y - gray.shape[0]/2)**2) / 100)
                size_score = min(area / 1000, 1.0)
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
                    "circularity": 1.0
                }
    
    except Exception as e:
        logger.error(f"Error in Hough circles detection: {str(e)}")
    
    return create_empty_result()

def detect_with_adaptive_threshold(gray):
    """Detect pupils using adaptive thresholding"""
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return find_best_ellipse(contours)
    
    except Exception as e:
        logger.error(f"Error in adaptive threshold detection: {str(e)}")
    
    return create_empty_result()

def detect_with_basic_threshold(gray):
    """Detect pupils using basic thresholding"""
    try:
        threshold_values = [20, 30, 40, 50, 60]
        
        for thresh_val in threshold_values:
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            result = find_best_ellipse(contours)
            if result["pupil_area"] > 0:
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
                
                area = np.pi * (MA / 2) * (ma / 2)
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                
                if (20 < area < 15000 and
                    circularity > 0.3 and
                    MA > 0 and ma > 0 and
                    (MA/ma) < 3):
                    
                    if area > best_area:
                        best_ellipse = ellipse
                        best_area = area
                        best_circularity = circularity
            
            except Exception as e:
                continue
    
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
    """Create empty result for single frame"""
    return {
        "pupil_area": 0,
        "center": None,
        "axes": None,
        "angle": None,
        "circularity": 0
    }

def create_empty_video_result():
    """Create empty result for video analysis"""
    return {
        "success": False,
        "video_info": {
            "duration": 0,
            "fps": 0,
            "total_frames": 0,
            "processed_frames": 0
        },
        "summary": {
            "average_area": 0,
            "min_area": 0,
            "max_area": 0,
            "std_area": 0,
            "area_range": 0,
            "detection_rate": 0
        },
        "pupil_data": []
    }