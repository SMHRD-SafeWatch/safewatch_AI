# detector.py
from ultralytics import YOLO
import cv2
from datetime import datetime, timedelta
import os
from database import insert_detection_data
from utils.detection_utils import (
    check_overlap,
    check_vertical_stack,
    check_irregular_stack
)
from utils.boundingbox_utils import draw_bounding_boxes, draw_status

class SafetyDetector:
    def __init__(self):
        self.CLASS_NAMES = ['human', 'hard_hat', 'safety_vest', 'box']
        self.COLORS = {
            'human': (255, 255, 255),
            'hard_hat': (0, 255, 0),
            'safety_vest': (0, 255, 255),
            'box': (0, 0, 255)
        }
        self.CLASS_CONF_THRESHOLDS = {
            'human': 0.8,
            'hard_hat': 0.7,
            'safety_vest': 0.7,
            'box': 0.85
        }
        self.model = YOLO('./best_final.pt')
        self.model.iou = 0.5
        self.last_capture_time = 0
        self.warning_delay = timedelta(seconds=30)  # 경고 알림 딜레이 시간 (30초)
        self.last_warning_time = datetime.now() - self.warning_delay
        self.save_dir = "captures"
        os.makedirs(self.save_dir, exist_ok=True)
            
    def process_detections(self, frame):
        results = self.model(frame, verbose=False)
        detections = {cls_name: [] for cls_name in self.CLASS_NAMES}

        # 탐지 결과를 딕셔너리에 추가
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                if cls >= len(self.CLASS_NAMES):
                    continue
                class_name = self.CLASS_NAMES[cls]
                
                # 클래스별 CONF 임계값 확인
                if conf >= self.CLASS_CONF_THRESHOLDS[class_name]:
                    detections[class_name].append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'conf': conf
                    })

        # 박스 상태 확인
        box_stack_status = "SAFE"
        risk_level = "SAFE"
        if check_vertical_stack(detections):  # 수직 스택 상태
            box_stack_status = "HIGH"
            risk_level = "MEDIUM"
        elif check_irregular_stack(detections):  # 불규칙 스택 확인
            box_stack_status = "IRREGULAR"
            risk_level = "HIGH"

        # 위험 상태 초기화
        near_box = False
        helmet_detected = False
        vest_detected = False

        # 사람이 감지되었을 때 추가 위험 평가 수행
        content = []
        if len(detections['human']) > 0:
            for person in detections['human']:
                px1, py1, px2, py2 = person['bbox']
                helmet_detected = any(
                    check_overlap((px1, py1, px2, py2), helmet['bbox'])
                    for helmet in detections['hard_hat']
                )
                vest_detected = any(
                    check_overlap((px1, py1, px2, py2), vest['bbox'])
                    for vest in detections['safety_vest']
                )
                near_box = any(
                    check_overlap(person['bbox'], box_det['bbox'])
                    for box_det in detections['box']
                )

                # 위험 상태 평가
                if box_stack_status == "HIGH":
                    if near_box:
                        content.append("Box Stack High and Near Box")
                        risk_level = "HIGH"
                    else:
                        content.append("Box Stack High")
                        risk_level = "MEDIUM"
                elif box_stack_status == "IRREGULAR":
                    content.append("Irregular Box Stack")
                    risk_level = "HIGH"
                if not helmet_detected:
                    content.append("No Helmet")
                if not vest_detected:
                    content.append("No Vest")
                if near_box:
                    content.append("Near Box")

        if not content and box_stack_status != "SAFE":
            content_text = box_stack_status
        elif not content:
            content_text = "All Safe"
        else:
            content_text = ", ".join(content)

        detection_info = {
            "camera_id": "CAM_002",
            "detection_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "detection_object": {
                "hard_hat": helmet_detected,
                "safety_vest": vest_detected,
                "near_box": near_box,
                "box_stack": box_stack_status
            },
            "risk_level": risk_level,
            "content": content_text
        }

        # 위험 상황을 데이터베이스에 저장
        self.save_risk_data(frame, risk_level, detection_info)

        # 상태 정보 표시
        status_texts = [
            (f"Risk Level: {risk_level}", (0, 0, 255) if risk_level == "HIGH" else (0, 255, 0)),
            (f"Box Stack: {box_stack_status}", (0, 0, 255) if box_stack_status == "HIGH" else (0, 255, 0)),
            (f"Near Box: {'Yes' if near_box else 'No'}", (0, 0, 255) if near_box else (0, 255, 0)),
            (f"Helmet: {'ON' if helmet_detected else 'OFF'}", (0, 255, 0) if helmet_detected else (0, 0, 255)),
            (f"Vest: {'ON' if vest_detected else 'OFF'}", (0, 255, 0) if vest_detected else (0, 0, 255))
        ]
        draw_status(frame, status_texts)

        # 바운딩 박스 그리기
        draw_bounding_boxes(frame, detections, self.COLORS)

        return detection_info

    def save_risk_data(self, frame, risk_level, detection_info):
        """위험 상황을 데이터베이스에 저장하고 이미지를 저장합니다."""
        current_time = datetime.now()
        if risk_level in ["HIGH", "MEDIUM"] and \
           (current_time - self.last_warning_time >= self.warning_delay):
            filename = f"{self.save_dir}/{current_time.strftime('%Y-%m-%d_%H-%M-%S')}_risk.jpg"
            cv2.imwrite(filename, frame)

            try:
                with open(filename, "rb") as img_file:
                    image_blob = img_file.read()
                insert_detection_data(
                    camera_id=detection_info["camera_id"],
                    detection_time=current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    detection_object=str(detection_info["detection_object"]),
                    image_url=image_blob,
                    risk_level=risk_level,
                    content=detection_info["content"]
                )
                print("Data inserted into database successfully.")
            except Exception as e:
                print(f"Failed to insert data into database: {e}")

            self.last_warning_time = current_time
