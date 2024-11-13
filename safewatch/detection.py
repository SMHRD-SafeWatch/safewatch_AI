from ultralytics import YOLO
import cv2
from datetime import datetime
import json

class SafetyDetector:
    def __init__(self, db_connection):
        self.CLASS_NAMES = ['human', 'hard_hat', 'safety_vest']
        self.COLORS = {
            'human': (255, 255, 255),
            'hard_hat': (0, 255, 0),
            'safety_vest': (0, 255, 255)
        }
        self.CONF_THRESHOLDS = {
            'human': 0.8,
            'hard_hat': 0.85,  
            'safety_vest': 0.8
        }
        self.model = YOLO('../models/best_final.pt')    
        self.model.conf = min(self.CONF_THRESHOLDS.values())
        self.model.iou = 0.5
        self.db = db_connection

    def check_overlap(self, region1, region2):
        """두 영역(bbox)의 겹침 여부를 확인"""
        x1, y1, x2, y2 = region1
        x3, y3, x4, y4 = region2
        return not ((x1 >= x4) or (x2 <= x3) or (y1 >= y4) or (y2 <= y3))

    def process_detections(self, frame):
        """프레임에서 객체를 탐지하고 안전장비 착용 여부를 확인"""
        results = self.model(frame, verbose=False)
        detections = {
            'human': [],
            'hard_hat': [],
            'safety_vest': []
        }
        
        # 객체 검출 로직
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls >= len(self.CLASS_NAMES):
                    continue
                
                class_name = self.CLASS_NAMES[cls]
                if class_name == 'box':
                    continue
                
                class_threshold = self.CONF_THRESHOLDS[class_name]
                if conf >= class_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections[class_name].append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'conf': conf
                    })
        
        detection_results = []
        
        # 각 사람별 처리
        for person in detections['human']:
            # 영역 계산
            px1, py1, px2, py2 = person['bbox']
            person_height = py2 - py1
            person_width = px2 - px1
            
            # 머리 영역 계산
            head_height = person_height // 7
            head_width = person_width // 2.6
            
            head_center_x = (px1 + px2) // 2
            head_x1 = int(head_center_x - head_width // 2)
            head_x2 = int(head_center_x + head_width // 2)
            
            head_region = (head_x1, py1, head_x2, py1 + head_height)
            
            # 몸통 영역 계산
            gap = person_height // 15
            
            body_width = int(person_width * 0.7)
            body_center_x = (px1 + px2) // 2
            body_x1 = int(body_center_x - body_width // 2)
            body_x2 = int(body_center_x + body_width // 2)
            
            body_start = py1 + head_height + gap
            body_region = (body_x1, body_start, body_x2, py2)
            
            # 안전장비 감지
            helmet_detected = False
            for helmet in detections['hard_hat']:
                hx1, hy1, hx2, hy2 = helmet['bbox']
                if self.check_overlap(head_region, (hx1, hy1, hx2, hy2)):
                    helmet_detected = True
                    break
            
            vest_detected = False
            for vest in detections['safety_vest']:
                vx1, vy1, vx2, vy2 = vest['bbox']
                if self.check_overlap(body_region, (vx1, vy1, vx2, vy2)):
                    vest_detected = True
                    break
            
            # 미착용 항목 저장
            undetected_items = []
            if not helmet_detected:
                undetected_items.append("hard_hat")
            if not vest_detected:
                undetected_items.append("safety_vest")
            
            # 위험 레벨 결정
            if not helmet_detected and not vest_detected:
                risk_level = "MEDIUM"
                content = "전부 미착용"
            elif not helmet_detected:
                risk_level = "LOW"
                content = "안전모 미착용"
            elif not vest_detected:
                risk_level = "LOW"
                content = "안전조끼 미착용"
            else:
                risk_level = "SAFE"
                content = "전부 착용"

            # 시각화
            cv2.rectangle(frame, (px1, py1), (px2, py2), self.COLORS['human'], 2)
            
            text_color = (0, 255, 0) if helmet_detected and vest_detected else (0, 0, 255)
            
            status_text = f"Safety Hat: {'OK' if helmet_detected else 'X'}"
            status_text += f" | Vest: {'OK' if vest_detected else 'X'}"
            
            cv2.putText(frame, status_text, (px1, py1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            cv2.putText(frame, f"Risk: {risk_level}", (px1, py1 - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            
            if helmet_detected:
                cv2.rectangle(frame, 
                            (head_region[0], head_region[1]),
                            (head_region[2], head_region[3]),
                            self.COLORS['hard_hat'], 2)
            if vest_detected:
                cv2.rectangle(frame, 
                            (body_region[0], body_region[1]),
                            (body_region[2], body_region[3]),
                            self.COLORS['safety_vest'], 2)

            current_time = datetime.now()
            
            detection_info = {
                "camera_id": "CAM_001",
                "detection_time": current_time,
                "detection_object": ",".join(undetected_items),  # 미착용 항목들을 쉼표로 구분된 문자열로 저장
                "risk_level": risk_level,
                "content": content
            }
            
            # SAFE가 아니고 미착용 항목이 있는 경우에만 DB에 저장
            if risk_level != "SAFE" and undetected_items and self.db is not None:
                try:
                    # 이미지를 바이너리 데이터로 변환
                    _, img_encoded = cv2.imencode('.jpg', frame)
                    img_bytes = img_encoded.tobytes()
                    
                    # DB에 저장
                    self.db.insert_detection(
                        camera_id=detection_info["camera_id"],
                        detection_time=detection_info["detection_time"],
                        detection_object=detection_info["detection_object"],
                        risk_level=detection_info["risk_level"],
                        content=detection_info["content"],
                        image_url=img_bytes
                    )
                    
                    print(f"Detection saved to DB at {current_time} - Undetected items: {undetected_items}")
                    
                except Exception as e:
                    print(f"Error saving to database: {e}")
            
            detection_results.append(detection_info)

        return detection_results