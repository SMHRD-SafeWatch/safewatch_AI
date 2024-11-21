import time
import os
from datetime import datetime
import cv2
from database import async_insert_detection_data
from models.model import get_model
import threading
from utils import zone
from utils.encode import image_encode
from utils import config

model = get_model()

captured_warning_zone = False
captured_danger_zone = False
last_capture_time_warning = 0
last_capture_time_danger = 0
capture_cooldown = 2  # 1초 대기

#모델 예측
def detect_objects(frame):
    results = model(frame,conf=0.75)
    return results

#감지 시 캡처, DB삽입
def hand_detections(results, display_frame):
    global captured_warning_zone, captured_danger_zone, last_capture_time_warning, last_capture_time_danger
    hand_in_warning_zone = False
    hand_in_danger_zone = False

    # 위험도가 더 높은 Danger Zone을 우선 처리하기 위해 플래그 추가
    danger_zone_detected = False
    
    for result in results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            label = int(box.cls[0].item())
            class_name = model.names[label]
            
            cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)
            label_text = f"{class_name}: {conf:.2f}"
            cv2.putText(display_frame, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if zone.is_inside_danger_zone((x_min, y_min, x_max, y_max), config.danger_zone_start, config.danger_zone_end):
                danger_zone_detected = True  # Danger Zone 감지
                hand_in_danger_zone = True
                cv2.putText(display_frame, 'WARNING : Danger_Zone', (550, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 128, 255), 2)
                cv2.putText(display_frame, 'RISK-LEVEL : HIGH', (550, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 128, 255), 2)
                if not captured_danger_zone and time.time() - last_capture_time_danger >= capture_cooldown:
                    timestamp, image_binary = image_encode(display_frame)
                    async_insert_detection_data("CAM_003", timestamp, class_name, image_binary, 'HIGH', "손이 위험구역에 감지됨.") # DB삽입
                    captured_danger_zone = True
                    last_capture_time_danger = time.time()
                    
            elif not danger_zone_detected and zone.is_inside_danger_zone((x_min, y_min, x_max, y_max), config.warning_zone_start, config.warning_zone_end):
                hand_in_warning_zone = True
                cv2.putText(display_frame, 'WARNING : Warning_Zone', (550, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
                cv2.putText(display_frame, 'RISK-LEVEL : MEDIUM', (550, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
                if not captured_warning_zone and time.time() - last_capture_time_warning >= capture_cooldown:
                    timestamp, image_binary = image_encode(display_frame)
                    async_insert_detection_data("CAM_003", timestamp, class_name, image_binary, 'MEDIUM', "손이 경고구역에 감지됨.") # DB삽입
                    captured_warning_zone = True
                    last_capture_time_warning = time.time()
                        
    # 손이 구역을 완전히 벗어난 경우 일정 시간 동안 캡처 방지
    if not hand_in_danger_zone and time.time() - last_capture_time_danger >= capture_cooldown:
        captured_danger_zone = False  # 일정 시간이 지난 후 캡처 상태 초기화
    if not hand_in_warning_zone and time.time() - last_capture_time_warning >= capture_cooldown:
        captured_warning_zone = False  # 일정 시간이 지난 후 캡처 상태 초기화                 