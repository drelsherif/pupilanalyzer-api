import cv2
import numpy as np
import logging
import tempfile
import os
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_pupil_response(image_bytes):
    """Analyze single image for pupil detection (legacy function)"""
    try:
        npimg = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if frame is None:
            logger.error("Failed to decode image")
            return create_empty_result()
        
        logger.info(f"Image decoded: {frame.shape}")
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

def analyze_pupil_video(video_bytes: bytes, test_metadata: dict = None) -> Dict[str, Any]:
    """
    Analyze pupil response from entire video file - ENHANCED VERSION for Light Response
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
        
        # ENHANCED: Dynamic frame processing based on test type and duration
        test_type = "general"
        if test_metadata:
            test_type = test_metadata.get("test_type", "general")
        
        # Determine optimal frame count for better light response detection
        if test_type == "light_response":
            # For light response: ensure good temporal resolution (at least 5 frames per second)
            min_frames_per_second = 5
            max_frames_to_process = min(int(duration * min_frames_per_second), 50)  # Cap at 50
            logger.info(f"LIGHT RESPONSE MODE: Using {max_frames_to_process} frames for enhanced detection")
        elif duration <= 3.0:
            # Short videos: moderate frame count
            max_frames_to_process = 20
            logger.info(f"SHORT VIDEO MODE: Using {max_frames_to_process} frames")
        elif duration >= 10.0:
            # Long videos: reasonable sampling
            max_frames_to_process = 40
            logger.info(f"LONG VIDEO MODE: Using {max_frames_to_process} frames")
        else:
            # Standard videos: balanced approach
            max_frames_to_process = 30
            logger.info(f"STANDARD MODE: Using {max_frames_to_process} frames")
        
        sample_interval = max(1, frame_count // max_frames_to_process)
        
        logger.info(f"ENHANCED: Processing every {sample_interval}th frame (max {max_frames_to_process} frames)")
        
        pupil_data = []
        frame_number = 0
        processed_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process selected frames
            if frame_number % sample_interval == 0 and processed_count < max_frames_to_process:
                timestamp = frame_number / fps if fps > 0 else 0
                
                # Use optimized detection method
                pupil_result = analyze_single_frame_fast(frame)
                
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
                    
                    logger.info(f"Frame {frame_number} (t={timestamp:.2f}s): pupil_area = {pupil_result['pupil_area']:.1f}")
                
                processed_count += 1
            
            frame_number += 1
        
        cap.release()
        
        # Calculate summary statistics
        if pupil_data:
            areas = [d["pupil_area"] for d in pupil_data]
            timestamps = [d["timestamp"] for d in pupil_data]
            
            summary = {
                "average_area": round(np.mean(areas), 2),
                "min_area": round(np.min(areas), 2),
                "max_area": round(np.max(areas), 2),
                "std_area": round(np.std(areas), 2),
                "area_range": round(np.max(areas) - np.min(areas), 2),
                "detection_rate": round(len(pupil_data) / processed_count * 100, 1) if processed_count > 0 else 0,
                # ENHANCED: Add light response specific metrics
                "temporal_resolution": round(len(pupil_data) / duration, 2) if duration > 0 else 0,
                "analysis_mode": test_type
            }
            
            # ENHANCED: Light response analysis
            if test_type == "light_response" and len(pupil_data) >= 3:
                light_response = analyze_light_response_pattern(pupil_data, duration)
                summary.update(light_response)
        else:
            summary = {
                "average_area": 0,
                "min_area": 0,
                "max_area": 0,
                "std_area": 0,
                "area_range": 0,
                "detection_rate": 0,
                "temporal_resolution": 0,
                "analysis_mode": test_type
            }
        
        result = {
            "success": len(pupil_data) > 0,
            "video_info": {
                "duration": round(duration, 2),
                "fps": round(fps, 2),
                "total_frames": frame_count,
                "processed_frames": len(pupil_data),
                "sample_interval": sample_interval,
                "test_type": test_type
            },
            "summary": summary,
            "pupil_data": pupil_data
        }
        
        logger.info(f"ENHANCED analysis complete: {len(pupil_data)} pupil detections from {processed_count} processed frames")
        if test_type == "light_response":
            logger.info(f"Light response temporal resolution: {summary.get('temporal_resolution', 0):.1f} frames/second")
        
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

def analyze_light_response_pattern(pupil_data: List[Dict], duration: float) -> Dict[str, Any]:
    """
    ENHANCED: Analyze pupil light response pattern for clinical metrics
    """
    try:
        areas = [d["pupil_area"] for d in pupil_data]
        timestamps = [d["timestamp"] for d in pupil_data]
        
        # Divide into phases based on timestamp (assuming 6-second test: 0-2s baseline, 2-4s light, 4-6s recovery)
        baseline_data = [area for area, t in zip(areas, timestamps) if 0 <= t < 2.0]
        light_data = [area for area, t in zip(areas, timestamps) if 2.0 <= t < 4.0]
        recovery_data = [area for area, t in zip(areas, timestamps) if 4.0 <= t <= 6.0]
        
        light_response_metrics = {}
        
        if baseline_data and light_data:
            baseline_avg = np.mean(baseline_data)
            light_min = np.min(light_data)
            
            # Calculate pupil constriction percentage
            constriction_percent = ((baseline_avg - light_min) / baseline_avg * 100) if baseline_avg > 0 else 0
            light_response_metrics["pupil_constriction_percent"] = round(constriction_percent, 2)
            light_response_metrics["baseline_average"] = round(baseline_avg, 2)
            light_response_metrics["minimum_during_light"] = round(light_min, 2)
            
            # Response quality assessment
            if constriction_percent > 20:
                response_quality = "Excellent"
            elif constriction_percent > 15:
                response_quality = "Good" 
            elif constriction_percent > 10:
                response_quality = "Moderate"
            elif constriction_percent > 5:
                response_quality = "Weak"
            else:
                response_quality = "Poor"
            
            light_response_metrics["response_quality"] = response_quality
            light_response_metrics["phase_data_quality"] = {
                "baseline_frames": len(baseline_data),
                "light_frames": len(light_data),
                "recovery_frames": len(recovery_data)
            }
            
            logger.info(f"Light Response Analysis: {constriction_percent:.1f}% constriction, Quality: {response_quality}")
        
        return light_response_metrics
        
    except Exception as e:
        logger.error(f"Error in light response analysis: {str(e)}")
        return {"light_response_error": str(e)}

def analyze_single_frame_fast(frame: np.ndarray) -> Dict[str, Any]:
    """
    OPTIMIZED: Analyze single frame using only the fastest method
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # OPTIMIZATION: Only use HoughCircles (fastest and most reliable)
        result = detect_with_hough_circles_fast(gray)
        if result["pupil_area"] > 0:
            return result
        
        # Fallback to basic threshold only if HoughCircles fails
        return detect_with_basic_threshold_fast(gray)
        
    except Exception as e:
        logger.error(f"Error analyzing frame: {str(e)}")
        return create_empty_result()

def detect_with_hough_circles_fast(gray):
    """OPTIMIZED: Faster HoughCircles detection"""
    try:
        # OPTIMIZATION: Smaller image for faster processing
        height, width = gray.shape
        if width > 640:
            scale = 640 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            gray = cv2.resize(gray, (new_width, new_height))
        
        # OPTIMIZATION: Less aggressive CLAHE
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4,4))
        enhanced = clahe.apply(gray)
        
        # OPTIMIZATION: Relaxed HoughCircles parameters for speed
        circles = cv2.HoughCircles(
            enhanced,
            cv2.HOUGH_GRADIENT,
            dp=2,  # Increased dp for speed
            minDist=20,  # Reduced minDist
            param1=40,  # Reduced param1
            param2=25,  # Reduced param2
            minRadius=8,
            maxRadius=100
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            # Take the first decent circle found
            for (x, y, r) in circles:
                area = np.pi * r * r
                if 30 < area < 8000:  # Reasonable pupil size
                    return {
                        "pupil_area": round(area, 2),
                        "center": [round(float(x), 2), round(float(y), 2)],
                        "axes": [round(float(r*2), 2), round(float(r*2), 2)],
                        "angle": 0.0,
                        "circularity": 1.0
                    }
    
    except Exception as e:
        logger.error(f"Error in fast Hough circles detection: {str(e)}")
    
    return create_empty_result()

def detect_with_basic_threshold_fast(gray):
    """OPTIMIZED: Faster basic threshold detection"""
    try:
        # OPTIMIZATION: Try only one threshold value
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)  # Smaller kernel
        _, thresh = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find largest reasonable contour
        for cnt in contours:
            if len(cnt) >= 5:
                try:
                    ellipse = cv2.fitEllipse(cnt)
                    (x, y), (MA, ma), angle = ellipse
                    
                    area = np.pi * (MA / 2) * (ma / 2)
                    
                    if (30 < area < 8000 and  # Reasonable size
                        MA > 0 and ma > 0 and
                        (MA/ma) < 2.5):  # Not too elongated
                        
                        return {
                            "pupil_area": round(area, 2),
                            "center": [round(float(x), 2), round(float(y), 2)],
                            "axes": [round(float(MA), 2), round(float(ma), 2)],
                            "angle": round(float(angle), 2),
                            "circularity": 0.8  # Assume decent circularity
                        }
                
                except Exception as e:
                    continue
    
    except Exception as e:
        logger.error(f"Error in fast basic threshold detection: {str(e)}")
    
    return create_empty_result()

# Keep original detection methods for single image analysis
def detect_with_hough_circles(gray):
    """Original HoughCircles detection for single images"""
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
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
    """Original adaptive thresholding"""
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
    """Original basic thresholding"""
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