import cv2
from typing import Iterator

def generate_frames_feed(camera) -> Iterator[bytes]:
    """Generate video frames for streaming"""
    while True:
        frame = camera.read_frame()
        if frame is None:
            break
            
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')