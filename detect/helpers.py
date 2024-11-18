# helpers.py
import cv2

def generate_frames_feed(camera, detector):
    while True:
        frame = camera.read_frame()
        if frame is None:
            break
        detector.process_detections(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
