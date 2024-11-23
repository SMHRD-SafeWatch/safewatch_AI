# helpers.py
import cv2
from concurrent.futures import ThreadPoolExecutor

def generate_frames_feed(camera, detector):
    with ThreadPoolExecutor(max_workers=2) as executor:
        while True:
            frame = camera.read_frame()
            if frame is None:
                continue  # 프레임이 없으면 다음 루프로 이동

            # 탐지 작업을 멀티스레드로 수행
            future = executor.submit(detector.process_detections, frame)
            future.result()

            # 프레임을 JPEG로 인코딩
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue  # 인코딩 실패 시 다음 루프로 이동

            # 스트리밍용 데이터 생성
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

