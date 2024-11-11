# tests/test_basic.py
import pytest
from safewatch.core import Camera, SafetyDetector

def test_camera():
    camera = Camera()
    frame = camera.read_frame()
    assert frame is not None
    assert frame.shape == (480, 640, 3)
    camera.release()

def test_detector():
    detector = SafetyDetector()
    assert detector is not None
    assert detector.model.conf == 0.8
    assert detector.model.iou == 0.5

def test_overlap_check():
    detector = SafetyDetector()
    region1 = (0, 0, 10, 10)
    region2 = (5, 5, 15, 15)
    region3 = (20, 20, 30, 30)
    
    assert detector._check_overlap(region1, region2)
    assert not detector._check_overlap(region1, region3)

# tests/test_advanced.py
from fastapi.testclient import TestClient
from safewatch.app import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200

def test_stream_endpoint():
    response = client.get("/stream")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]