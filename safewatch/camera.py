import cv2
from typing import Optional

class Camera:
    def __init__(self):
        """Initialize camera with default settings"""
        self.camera = cv2.VideoCapture(0)
        self._setup_camera()
    
    def _setup_camera(self):
        """Setup camera properties"""
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 60)
    
    def read_frame(self) -> Optional[cv2.Mat]:
        """Read a frame from camera"""
        success, frame = self.camera.read()
        if success:
            return cv2.resize(frame, (640, 480))
        return None
    
    def release(self):
        """Release camera resources"""
        self.camera.release()