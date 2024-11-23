# detoctor.py
from ultralytics import YOLO
import cv2
from datetime import datetime, timedelta
from utils.database import async_insert_detection_data
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
            'box': 0.9
        }
        self.model = YOLO('models/best.pt')
        self.model.to('cuda') # GPU 사용 설정
        self.model.iou = 0.5
        self.warning_delay = timedelta(seconds=30)  # 경고 알림 딜레이 시간
        self.last_warning_time = datetime.now() - self.warning_delay

        # 박스 상태 유지 관련 초기화
        self.last_box_stack_status = "SAFE"  # 마지막 박스 상태 저장
        self.box_status_timeout = timedelta(seconds=10)  # 박스 상태 유지 시간
        self.last_box_detected_time = datetime.now()
            
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
        current_time = datetime.now()
        box_stack_status = "SAFE"
        if detections['box']:
            if check_vertical_stack(detections):  # 수직 스택 상태
                box_stack_status = "HIGH"
            elif check_irregular_stack(detections):  # 불규칙 스택 확인
                box_stack_status = "IRREGULAR"
            self.last_box_stack_status = box_stack_status
            self.last_box_detected_time = current_time
        else:
            time_since_last_box = current_time - self.last_box_detected_time
            if time_since_last_box <= self.box_status_timeout:
                box_stack_status = self.last_box_stack_status
            else:
                box_stack_status = "UNKNOWN"
                self.last_box_stack_status = "UNKNOWN"

        # 위험 수준 결정
        risk_level = "SAFE"
        if box_stack_status == "HIGH":
            risk_level = "MEDIUM"
        elif box_stack_status == "IRREGULAR":
            risk_level = "HIGH"

        # 위험 상태 초기화
        near_box = False
        helmet_detected = False
        vest_detected = False
        content = []

        # 사람이 감지되었을 때 추가 위험 평가 수행
        if detections['human']:
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
                elif box_stack_status == "UNKNOWN":
                    content.append("Box Status Unknown")
                    risk_level = "MEDIUM"

                if not helmet_detected:
                    content.append("No Helmet")
                if not vest_detected:
                    content.append("No Vest")
                if near_box:
                    content.append("Near Box")
        else:
            # 사람이 없는 경우 박스 상태만 평가
            if box_stack_status == "HIGH":
                content.append("Box Stack High")
                risk_level = "MEDIUM"
            elif box_stack_status == "IRREGULAR":
                content.append("Irregular Box Stack")
                risk_level = "HIGH"
            elif box_stack_status == "UNKNOWN":
                content.append("Box Status Unknown")
                risk_level = "HIGH"

        # 최종 content_text 생성
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

        draw_bounding_boxes(frame, detections, self.COLORS)

        return detection_info

    def save_risk_data(self, frame, risk_level, detection_info):
        """위험 상황을 데이터베이스에 저장합니다."""
        current_time = datetime.now()
        
        # 데이터 삽입 제어 확인
        from utils.database import DB_INSERT_ENABLED  # DB 삽입 플래그 확인
        if not DB_INSERT_ENABLED:
            # 딜레이 로직 추가
            if current_time - self.last_warning_time < timedelta(seconds=30):
                return
            print("DB 삽입 연결 X")
            self.last_warning_time = current_time 
            return
        
        if risk_level in ["HIGH", "MEDIUM"] and \
           (current_time - self.last_warning_time >= self.warning_delay):
            try:
                _, img_encoded = cv2.imencode('.jpg', frame)
                image_blob = img_encoded.tobytes()
                async_insert_detection_data(
                    camera_id=detection_info["camera_id"],
                    detection_time=current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    detection_object=str(detection_info["detection_object"]),
                    image_url=image_blob,
                    risk_level=risk_level,
                    content=detection_info["content"]
                )
                print("DB 데이터 삽입 성공")
            except Exception as e:
                print(f"DB 데이터 삽입 실패 {e}")

            self.last_warning_time = current_time
