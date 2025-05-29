from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from process import analyze_pupil_response, analyze_pupil_video
import logging

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

@app.get("/")
async def root():
    return {
        "message": "Pupil Analyzer API v2.0",
        "endpoints": {
            "/analyze": "POST - Analyze single image",
            "/analyze-video": "POST - Analyze entire video (recommended)"
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
        logger.info(f"Analyzing video: {file.filename}, size: {file.size} bytes")
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('video/'):
            logger.warning(f"Non-video file uploaded: {file.content_type}")
            # Still try to process it, might be a video with wrong MIME type
        
        contents = await file.read()
        result = analyze_pupil_video(contents)
        
        logger.info(f"Video analysis complete: {result['success']}")
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "2.0.0"}