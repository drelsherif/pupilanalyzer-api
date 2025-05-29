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
    Returns time-series data with statistics
    """
    try:
        logger.info(f"Analyzing pupil video: {file.filename}, size: {file.size} bytes")
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('video/'):
            logger.warning(f"Non-video file uploaded: {file.content_type}")
            # Still try to process it, might be a video with wrong MIME type
        
        contents = await file.read()
        result = analyze_pupil_video(contents)
        
        logger.info(f"Pupil video analysis complete: {result['success']}")
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing pupil video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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