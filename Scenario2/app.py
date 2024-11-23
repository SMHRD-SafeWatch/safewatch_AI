# app.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from utils.camera import Camera
from utils.detector import SafetyDetector
import uvicorn
from utils.helpers import generate_frames_feed

app = FastAPI()

try:
    camera = Camera()  # 멀티스레드 카메라 초기화
except Exception as e:
    print(f"카메라 초기화 실패: {e}")
    raise RuntimeError("카메라 초기화에 실패했습니다. 애플리케이션을 시작할 수 없습니다.")

detector = SafetyDetector()

@app.get("/scenario2")
async def process_detection():
    if camera is None:
        return {"status": "error", "message": "카메라가 초기화되지 않았습니다."}
    
    frame = camera.read_frame()
    if frame is not None:
        detection_results = detector.process_detections(frame)
        if detection_results:
            return {"status": "success", "data": detection_results}
    return {"status": "error", "message": "탐지 결과가 없습니다."}

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames_feed(camera, detector),
                             media_type='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)