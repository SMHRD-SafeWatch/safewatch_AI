# camera.py
import cv2

class Camera:
    def __init__(self):
        self.camera = cv2.VideoCapture(1)
        self.setup_camera()
        
    def setup_camera(self):
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 60)

    def read_frame(self):
        success, frame = self.camera.read()
        if success:
            return cv2.resize(frame, (640, 480))
        return None
    
    def release(self):
        self.camera.release()
