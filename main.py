from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from process import analyze_pupil_response, analyze_pupil_video
from eye_movement_analysis import EyeMovementAnalyzer
import logging
import json
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Pupil Analyzer API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize eye movement analyzer
eye_movement_analyzer = EyeMovementAnalyzer()

@app.get("/")
async def root():
    return {
        "message": "PupilIO Analysis API v2.0",
        "services": ["Pupil Analysis", "Eye Movement Analysis"],
        "endpoints": {
            "/analyze": "POST - Analyze single image (legacy)",
            "/analyze-video": "POST - Analyze pupil video (recommended)",
            "/analyze-eye-movement": "POST - Analyze eye movement video (NEW)",
            "/health": "GET - Health check"
        }
    }

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze single image for pupil detection
    Legacy endpoint - use /analyze-video for better results
    """
    try:
        logger.info(f"Analyzing single image: {file.filename}")
        contents = await file.read()
        result = analyze_pupil_response(contents)
        return result
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    """
    Analyze entire video for pupil detection over time
    Returns time-series data with statistics - iOS Compatible Format
    """
    try:
        logger.info(f"Analyzing pupil video: {file.filename}, size: {file.size} bytes")
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('video/'):
            logger.warning(f"Non-video file uploaded: {file.content_type}")
        
        contents = await file.read()
        result = analyze_pupil_video(contents)
        
        logger.info(f"Pupil video analysis complete: {result['success']}")
        
        # Return in exact format iOS expects
        if result['success']:
            # Convert all numeric values to strings for iOS compatibility
            video_info = result["video_info"]
            summary = result["summary"] 
            pupil_data = result["pupil_data"]
            
            # Ensure all values are properly formatted for iOS parsing
            response = {
                "success": True,  # Boolean for iOS
                "video_info": {
                    "duration": str(video_info["duration"]),
                    "fps": video_info["fps"],  # Keep as number
                    "total_frames": video_info["total_frames"],  # Keep as number
                    "processed_frames": video_info["processed_frames"]  # Keep as number
                },
                "summary": {
                    "average_area": str(summary["average_area"]),
                    "min_area": str(summary["min_area"]),
                    "max_area": str(summary["max_area"]),
                    "std_area": str(summary["std_area"]),
                    "area_range": str(summary["area_range"]),
                    "detection_rate": summary["detection_rate"]  # Keep as number
                },
                "pupil_data": []
            }
            
            # Format pupil data with proper types
            for data_point in pupil_data:
                formatted_point = {
                    "pupil_area": str(data_point["pupil_area"]),
                    "timestamp": str(data_point["timestamp"]) if isinstance(data_point["timestamp"], (int, float)) else data_point["timestamp"],
                    "frame_number": data_point["frame_number"],
                    # Include additional fields iOS might expect
                    "center": data_point.get("center", [0, 0]),
                    "axes": data_point.get("axes", [0, 0]), 
                    "angle": data_point.get("angle", 0),
                    "circularity": data_point.get("circularity", 1.0)
                }
                response["pupil_data"].append(formatted_point)
            
            logger.info(f"Returning {len(response['pupil_data'])} pupil data points to iOS")
            return response
            
        else:
            return {
                "success": False,
                "error": "Analysis failed - no pupil data detected",
                "video_info": {
                    "duration": "0",
                    "fps": 0,
                    "total_frames": 0,
                    "processed_frames": 0
                },
                "summary": {
                    "average_area": "0",
                    "min_area": "0", 
                    "max_area": "0",
                    "std_area": "0",
                    "area_range": "0",
                    "detection_rate": 0
                },
                "pupil_data": []
            }
        
    except Exception as e:
        logger.error(f"Error analyzing pupil video: {str(e)}")
        return {
            "success": False,
            "error": f"Server error: {str(e)}",
            "video_info": {
                "duration": "0",
                "fps": 0,
                "total_frames": 0, 
                "processed_frames": 0
            },
            "summary": {
                "average_area": "0",
                "min_area": "0",
                "max_area": "0", 
                "std_area": "0",
                "area_range": "0",
                "detection_rate": 0
            },
            "pupil_data": []
        }

@app.post("/analyze-eye-movement")
async def analyze_eye_movement(
    file: UploadFile = File(...),
    test_data: Optional[str] = Form(None)
):
    """
    NEW: Analyze eye movement video for H-pattern tracking
    Returns movement analysis with smoothness, accuracy, and coordination scores
    """
    try:
        logger.info(f"Analyzing eye movement video: {file.filename}, size: {file.size} bytes")
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('video/'):
            logger.warning(f"Non-video file uploaded for eye movement: {file.content_type}")
            # Still try to process it
        
        # Parse test metadata if provided
        test_metadata = None
        if test_data:
            try:
                test_metadata = json.loads(test_data)
                logger.info(f"Test metadata received: {test_metadata}")
            except json.JSONDecodeError:
                logger.warning("Invalid test_data JSON, proceeding without metadata")
        
        contents = await file.read()
        result = eye_movement_analyzer.analyze_video(contents, test_metadata)
        
        logger.info(f"Eye movement analysis complete: {result['success']}")
        return {
            'success': True,
            'analysis_results': result,
            'message': 'Eye movement analysis completed successfully'
        }
        
    except Exception as e:
        logger.error(f"Error analyzing eye movement: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze eye movement: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "version": "2.0.0",
        "services": {
            "pupil_analysis": "active",
            "eye_movement_analysis": "active"
        }
    }

@app.get("/test-endpoint")
async def test_endpoint():
    """Test endpoint to verify server capabilities"""
    return {
        "message": "PupilIO Analysis Server is running",
        "endpoints": {
            "pupil_analysis": "/analyze-video (POST)",
            "eye_movement_analysis": "/analyze-eye-movement (POST)", 
            "health": "/health (GET)",
            "test": "/test-endpoint (GET)"
        },
        "requirements": {
            "video_file": "Required - video file in form-data",
            "test_data": "Optional - JSON metadata in form-data (eye movement only)",
            "supported_formats": [".mov", ".mp4", ".avi"]
        },
        "capabilities": [
            "Pupil size detection and tracking",
            "Light response analysis", 
            "Eye movement pattern analysis",
            "H-pattern accuracy scoring",
            "Movement smoothness assessment",
            "Eye coordination evaluation"
        ]
    }