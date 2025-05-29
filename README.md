# Pupil Analyzer API

This API detects and analyzes pupil area from uploaded eye images using OpenCV and FastAPI.

## Endpoint

**POST** `/analyze`  
Form field: `file` (image file)

Returns:
```json
{ "pupil_area": 123.45 }
