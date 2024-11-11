from ultralytics import YOLO
import cv2
from datetime import datetime
import os

class SafetyDetector:
    def __init__(self):
        # 기존 초기화 코드 유지
        self.CLASS_NAMES = ['human', 'hard_hat', 'safety_vest']
        self.COLORS = {
            'human': (255, 255, 255),
            'hard_hat': (0, 255, 0),
            'safety_vest': (0, 255, 255)
        }
        self.model = YOLO('model_5/best_full.pt')
        self.model.conf = 0.8
        self.model.iou = 0.5
        self.last_capture_time = 0

    def check_overlap(self, region1, region2):
        x1, y1, x2, y2 = region1
        x3, y3, x4, y4 = region2
        return not ((x1 >= x4) or (x2 <= x3) or (y1 >= y4) or (y2 <= y3))

    def process_detections(self, frame):
        results = self.model(frame, verbose=False)
        detections = {
            'human': [],
            'hard_hat': [],
            'safety_vest': []
        }
        
        # 검출된 객체들을 종류별로 분류
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                if cls >= len(self.CLASS_NAMES):
                    continue
                
                class_name = self.CLASS_NAMES[cls]
                if class_name == 'box':
                    continue

                if class_name in detections:
                    detections[class_name].append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'conf': conf
                    })
        
        detection_results = []
        
        # 각 사람별로 안전장비 착용 여부 확인
        for person in detections['human']:
            px1, py1, px2, py2 = person['bbox']
            person_height = py2 - py1
            person_width = px2 - px1
            
            head_height = person_height // 3
            head_width = person_width // 1.5
            
            head_center_x = (px1 + px2) // 2
            head_x1 = int(head_center_x - head_width // 2)
            head_x2 = int(head_center_x + head_width // 2)
            
            head_region = (head_x1, py1, head_x2, py1 + head_height)
            body_region = (px1, py1 + head_height, px2, py2)
            
            # 안전모 착용 확인
            helmet_detected = False
            for helmet in detections['hard_hat']:
                hx1, hy1, hx2, hy2 = helmet['bbox']
                if self.check_overlap(head_region, (hx1, hy1, hx2, hy2)):
                    helmet_detected = True
                    break
            
            # 안전조끼 착용 확인
            vest_detected = False
            for vest in detections['safety_vest']:
                vx1, vy1, vx2, vy2 = vest['bbox']
                if self.check_overlap(body_region, (vx1, vy1, vx2, vy2)):
                    vest_detected = True
                    break
            
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

            # 바운딩 박스와 텍스트 그리기
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
            
            if (current_time.timestamp() - self.last_capture_time >= 10 and 
                risk_level != "SAFE"):
                
                filename = f"captures/{current_time.strftime('%Y-%m-%d_%H_%M_%S')}_{risk_level}.jpg"
                os.makedirs('captures', exist_ok=True)
                cv2.imwrite(filename, frame)
                
                detection_info = {
                    "camera_id": "CAM_001",
                    "detection_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "detection_object": {
                        "hard_hat": helmet_detected,
                        "safety_vest": vest_detected
                    },
                    "image_url": filename,
                    "risk_level": risk_level,
                    "content": content
                }
                                
                self.last_capture_time = current_time.timestamp()

        return detection_results