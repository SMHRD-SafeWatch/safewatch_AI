import pytest
import cv2
import numpy as np
from safewatch.camera import Camera
from safewatch.detection import SafetyDetector

# 테스트용 fixtures
@pytest.fixture
def sample_frame():
    """테스트용 더미 프레임 생성"""
    return np.zeros((480, 640, 3), dtype=np.uint8)

@pytest.fixture
def mock_db():
    """테스트용 mock 데이터베이스"""
    class MockDB:
        def insert_detection(self, *args, **kwargs):
            pass
        def close(self):
            pass
    return MockDB()

# 카메라 테스트
def test_camera_initialization():
    camera = Camera()
    try:
        assert camera is not None
        frame = camera.read_frame()
        assert frame is not None
        assert frame.shape == (480, 640, 3)  # 설정된 해상도 확인
    finally:
        camera.release()

# 디텍터 테스트
def test_detector_initialization(mock_db):
    detector = SafetyDetector(db_connection=mock_db)
    assert detector is not None
    assert detector.model is not None
    assert len(detector.CLASS_NAMES) == 3
    assert detector.model.conf == min(detector.CONF_THRESHOLDS.values())
    assert detector.model.iou == 0.5

# 겹침 검사 테스트
def test_overlap_check(mock_db):
    detector = SafetyDetector(db_connection=mock_db)
    # 완전히 겹치는 경우
    assert detector.check_overlap((0, 0, 10, 10), (0, 0, 10, 10)) == True
    
    # 부분적으로 겹치는 경우
    assert detector.check_overlap((0, 0, 10, 10), (5, 5, 15, 15)) == True
    
    # 겹치지 않는 경우
    assert detector.check_overlap((0, 0, 10, 10), (20, 20, 30, 30)) == False

# 객체 탐지 프로세스 테스트
def test_process_detections(mock_db, sample_frame):
    detector = SafetyDetector(db_connection=mock_db)
    results = detector.process_detections(sample_frame, save_to_db=False)
    assert isinstance(results, list)