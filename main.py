from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from process import analyze_pupil_response

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    result = analyze_pupil_response(contents)
    return result
