from fastapi import FastAPI
from camera import Camera
from detection import SafetyDetector
from db_config import OracleDB
import uvicorn
import asyncio
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수
detection_running = False
latest_detection_result = None
latest_detection_time = None

# DB 연결 초기화
try:
    db = OracleDB()
except Exception as e:
    print(f"Database connection failed: {e}")
    db = None

try:
    camera = Camera()
except Exception as e:
    camera = None
    print(f"Camera initialization failed: {e}")

# detector 초기화 시 db 연결 전달
detector = SafetyDetector(db) if db is not None else None

async def continuous_detection():
    """10초마다 객체 탐지를 수행하는 백그라운드 태스크"""
    global detection_running, latest_detection_result, latest_detection_time
    
    while detection_running:
        if camera is None or detector is None:
            print("Camera or detector not initialized")
            await asyncio.sleep(10)
            continue
            
        try:
            frame = camera.read_frame()
            if frame is not None:
                # save_to_db=True로 설정하여 10초마다만 DB에 저장
                detection_results = detector.process_detections(frame, save_to_db=True)
                latest_detection_result = detection_results
                latest_detection_time = datetime.now()
                print(f"Detection completed at {latest_detection_time}")
            
        except Exception as e:
            print(f"Error during detection: {e}")
            
        await asyncio.sleep(10)

@app.on_event("startup")
async def startup_event():
    """앱 시작 시 객체 탐지 태스크 시작"""
    global detection_running
    detection_running = True
    asyncio.create_task(continuous_detection())

@app.on_event("shutdown")
async def shutdown_event():
    """앱 종료 시 정리 작업"""
    global detection_running
    detection_running = False
    if db:
        db.close()
    if camera:
        camera.release()

@app.get("/start_detection")
async def start_detection():
    """객체 탐지 시작"""
    global detection_running
    if not detection_running:
        detection_running = True
        asyncio.create_task(continuous_detection())
        return {"status": "success", "message": "Detection started"}
    return {"status": "info", "message": "Detection already running"}

@app.get("/stop_detection")
async def stop_detection():
    """객체 탐지 중지"""
    global detection_running
    if detection_running:
        detection_running = False
        return {"status": "success", "message": "Detection stopped"}
    return {"status": "info", "message": "Detection not running"}

@app.get("/latest_result")
async def get_latest_result():
    """가장 최근 탐지 결과 조회"""
    if latest_detection_result is None:
        return {
            "status": "error",
            "message": "No detection results available yet"
        }
    
    return {
        "status": "success",
        "data": latest_detection_result,
        "timestamp": latest_detection_time
    }

@app.get("/status")
async def get_status():
    """현재 탐지 상태 조회"""
    return {
        "detection_running": detection_running,
        "latest_detection_time": latest_detection_time,
        "camera_status": "connected" if camera is not None else "disconnected",
        "detector_status": "initialized" if detector is not None else "not initialized"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)