# camera.py
import cv2
import threading
from queue import Queue

class Camera:
    def __init__(self, camera_id=1):
        self.camera = cv2.VideoCapture(camera_id)
        self.frame_queue = Queue(maxsize=1)
        self.running = True
        self.setup_camera()
        self.thread = threading.Thread(target=self._read_frames, daemon=True)
        self.thread.start()

    def setup_camera(self):
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)

    def _read_frames(self):
        while self.running:
            success, frame = self.camera.read()
            if success and not self.frame_queue.full():
                self.frame_queue.put(cv2.resize(frame, (640, 480)))

    def read_frame(self):
        if not self.frame_queue.empty():
            return self.frame_queue.get()
        return None

    def release(self):
        self.running = False
        self.thread.join()
        self.camera.release()
