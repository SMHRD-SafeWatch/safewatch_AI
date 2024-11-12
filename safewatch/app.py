from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
from camera import Camera
from detection import SafetyDetector
import uvicorn

app = FastAPI()

try:
    camera = Camera()
except Exception as e:
    camera = None
    print(f"Camera initialization failed: {e}")

detector = SafetyDetector()

@app.get("/scenario1")  # POST를 GET으로 변경
async def process_detection():
    if camera is None:
        return {"status": "error", "message": "Camera not initialized"}
    
    frame = camera.read_frame()
    if frame is not None:
        detection_results = detector.process_detections(frame)
        if detection_results:
            return {"status": "success", "data": detection_results}
    return {"status": "error", "message": "No detection results"}
 
if __name__ == "__main__":
   
   uvicorn.run(app, host="0.0.0.0", port=8000)
