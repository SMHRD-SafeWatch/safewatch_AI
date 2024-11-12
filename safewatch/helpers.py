import cv2
from typing import Iterator

def generate_frames_feed(camera) -> Iterator[bytes]:
    """Generate video frames for streaming"""
    while True:
        try:
            frame = camera.read_frame()
            if frame is None:
                print("No frame received from the camera.")
                break
            
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Failed to encode the frame.")
                continue
            
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Error while reading frame from camera: {e}")
            break