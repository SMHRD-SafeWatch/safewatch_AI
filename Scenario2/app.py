# app.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from camera import Camera
from detector import SafetyDetector
import uvicorn
from helpers import generate_frames_feed

app = FastAPI()

try:
    camera = Camera()
except Exception as e:
    camera = None
    print(f"Camera initialization failed: {e}")

detector = SafetyDetector()

@app.get("/scenario2")
async def process_detection():
    if camera is None:
        return {"status": "error", "message": "Camera not initialized"}
    
    frame = camera.read_frame()
    if frame is not None:
        detection_results = detector.process_detections(frame)
        if detection_results:
            return {"status": "success", "data": detection_results}
    return {"status": "error", "message": "No detection results"}

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames_feed(camera, detector),
                             media_type='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
   uvicorn.run(app, host="0.0.0.0", port=8000)
