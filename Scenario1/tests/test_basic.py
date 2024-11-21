import pytest
import numpy as np
from safewatch.camera import Camera
from safewatch.detection import SafetyDetector
from safewatch.util.check_overlap import check_overlap 

@pytest.fixture
def sample_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)

@pytest.fixture
def mock_db():
    class MockDB:
        def insert_detection(self, *args, **kwargs):
            pass
        def close(self):
            pass
    return MockDB()

@pytest.fixture
def mock_camera(monkeypatch):
    class MockCamera:
        def __init__(self, index=0):
            self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
            self._is_opened = True
        
        def read(self):
            return True, self.frame
            
        def release(self):
            self._is_opened = False
            
        def isOpened(self):
            return self._is_opened
            
        def set(self, *args):
            pass
    
    import cv2
    monkeypatch.setattr(cv2, "VideoCapture", MockCamera)

def test_camera_initialization(mock_camera):
    camera = Camera()
    frame = camera.read_frame()
    assert frame is not None
    assert isinstance(frame, np.ndarray)
    assert len(frame.shape) == 3
    camera.release()

def test_detector_initialization(mock_db):
    detector = SafetyDetector(db_connection=mock_db)
    assert detector is not None
    assert detector.model is not None
    assert len(detector.CLASS_NAMES) == 3
    assert detector.model.conf == min(detector.CONF_THRESHOLDS.values())
    assert detector.model.iou == 0.5

def test_overlap_check():
    assert check_overlap((0, 0, 10, 10), (0, 0, 10, 10)) == True
    assert check_overlap((0, 0, 10, 10), (5, 5, 15, 15)) == False  # 50% 미만
    assert check_overlap((0, 0, 10, 10), (20, 20, 30, 30)) == False
    assert check_overlap((0, 0, 10, 10), (0, 0, 8, 8)) == True  # 64% 겹침

def test_process_detections(mock_db, sample_frame):
    detector = SafetyDetector(db_connection=mock_db)
    results = detector.process_detections(sample_frame, save_to_db=False)
    assert isinstance(results, list)
    assert len(results) == 0

def test_detector_confidence_thresholds(mock_db):
    detector = SafetyDetector(db_connection=mock_db)
    assert all(0 <= conf <= 1 for conf in detector.CONF_THRESHOLDS.values())

def test_camera_release(mock_camera):
    camera = Camera()
    camera.release()
    assert not camera.camera.isOpened()