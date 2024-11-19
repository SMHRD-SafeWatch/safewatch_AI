import cv2
from typing import Iterator
from safewatch.camera import Camera
from safewatch.detection import SafetyDetector


class StreamHandler:
    _instance = None

    def __new__(cls, db_connection=None):
        if cls._instance is None:
            cls._instance = super(StreamHandler, cls).__new__(cls)
            cls._instance.camera = Camera()
            cls._instance.detector = SafetyDetector(db_connection=db_connection)
            cls._instance.frame = None
        return cls._instance

    def read_frame(self):
        """카메라에서 프레임을 읽어오는 메서드"""
        if self.camera is None:
            return None
        return self.camera.read_frame()

    def generate_frames(self) -> Iterator[bytes]:
        """스트리밍용 프레임 생성기"""
        while True:
            frame = self.read_frame()
            if frame is None:
                continue
                
            # 바운딩 박스 표시용 detection (save_to_db=False)
            if self.detector is not None:
                try:
                    self.detector.process_detections(frame, save_to_db=False)
                except Exception as e:
                    print(f"Error during detection: {e}")
                
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    def cleanup(self):
        if self.camera is not None:
            self.camera.release()