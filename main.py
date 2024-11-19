from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from safewatch.db_config import OracleDB
import uvicorn
import asyncio
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from safewatch.util.stream import StreamHandler
from concurrent.futures import ThreadPoolExecutor
from functools import partial

app = FastAPI()

templates = Jinja2Templates(directory="safewatch/templates")

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

thread_pool = ThreadPoolExecutor(max_workers=3)

async def continuous_detection():
    """10초마다 객체 탐지를 수행하는 백그라운드 태스크"""
    global detection_running, latest_detection_result, latest_detection_time
    
    while detection_running:
        if not hasattr(app.state, 'stream_handler'):
            print("StreamHandler not initialized")
            await asyncio.sleep(5)
            continue
            
        try:
            # 스트리밍과 독립적으로 프레임 읽기
            frame = app.state.stream_handler.read_frame()
            if frame is not None:
                # Thread Pool 활용 detection 처리
                loop = asyncio.get_event_loop()
                # DB 저장을 위한 detection (save_to_db=True) >> ThreadPoolExecutor 활용 처리 개선
                detection_results = await loop.run_in_executor(thread_pool, partial(
                    app.state.stream_handler.detector.process_detections,frame,save_to_db=True)
                )         
                # detection_results = app.state.stream_handler.detector.process_detections(frame, save_to_db=True)
                latest_detection_result = detection_results
                latest_detection_time = datetime.now()
                print(f"Detection completed at {latest_detection_time}")
            
        except Exception as e:
            print(f"Error during detection: {e}")
        # 객체 탐지 루프 10초 
        await asyncio.sleep(10)
 
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/stream", response_class=HTMLResponse)
async def stream(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.on_event("startup")
async def startup_event():
    """앱 시작 시 객체 탐지 태스크 시작"""
    app.state.stream_handler = StreamHandler(db_connection=db)
    global detection_running
    detection_running = True
    asyncio.create_task(continuous_detection())

@app.on_event("shutdown")
async def shutdown_event():
    if hasattr(app.state, 'stream_handler'):
        app.state.stream_handler.cleanup()
    thread_pool.shutdown(wait=True)

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
    
@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        app.state.stream_handler.generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
    
@app.get("/status")
async def get_status():
    """현재 탐지 상태 조회"""
    stream_handler = getattr(app.state, 'stream_handler', None)
    camera_status = "connected" if (stream_handler and stream_handler.camera) else "disconnected"
    detector_status = "initialized" if (stream_handler and stream_handler.detector) else "not initialized"
    
    return {
        "detection_running": detection_running,
        "latest_detection_time": latest_detection_time,
        "camera_status": camera_status,
        "detector_status": detector_status
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)