from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
from camera import Camera
from detection import SafetyDetector

class DetectionData(BaseModel):
    camera_id: str
    detection_time: str
    detection_object: Dict[str, bool]
    image_url: str
    risk_level: str
    content: str

app = FastAPI()
camera = Camera()
detector = SafetyDetector()

@app.post("/")  # /detection 대신 / 사용
async def process_detection():
    frame = camera.read_frame()
    if frame is not None:
        detection_results = detector.process_detections(frame)
        if detection_results:
            return {"status": "success", "data": detection_results[0]}
    return {"status": "error", "message": "No detection results"}

@app.on_event("shutdown")
async def shutdown_event():
    camera.release()