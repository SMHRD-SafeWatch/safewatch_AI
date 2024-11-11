from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from camera import Camera
from detection import SafetyDetector
from helpers import generate_frames_feed
import uvicorn

app = FastAPI()
camera = Camera()
detector = SafetyDetector()

@app.get("/")
async def root():
    """Root endpoint for detection results"""
    frame = camera.read_frame()
    if frame is not None:
        detection_results = detector.process_detections(frame)
        if detection_results:
            return detection_results[0]
    return {"message": "No detection results"}

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    camera.release()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)