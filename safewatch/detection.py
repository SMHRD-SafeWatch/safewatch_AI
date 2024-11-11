from ultralytics import YOLO
import cv2
from datetime import datetime
import os
from typing import Dict, List, Tuple, Optional

class SafetyDetector:
    CLASS_NAMES = ['human', 'hard_hat', 'safety_vest']
    COLORS = {
        'human': (255, 255, 255),
        'hard_hat': (0, 255, 0),
        'safety_vest': (0, 255, 255)
    }
    
    def __init__(self, model_path: str = None):
        """Initialize detector with YOLO model"""
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                    'models', 'best_full.pt')
        
        self.model = YOLO(model_path)
        self.model.conf = 0.8
        self.model.iou = 0.5
        self.last_capture_time = 0
    
    def process_detections(self, frame) -> List[Dict]:
        """Process frame and detect safety equipment"""
        results = self.model(frame, verbose=False)
        detections = self._process_yolo_results(results)
        return self._analyze_detections(frame, detections)
    
    def _process_yolo_results(self, results) -> Dict:
        """Process YOLO detection results"""
        detections = {class_name: [] for class_name in self.CLASS_NAMES}
        
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls >= len(self.CLASS_NAMES):
                    continue
                
                class_name = self.CLASS_NAMES[cls]
                if class_name == 'box':
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections[class_name].append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'conf': float(box.conf[0])
                })
        
        return detections
    
    def _analyze_detections(self, frame, detections: Dict) -> List[Dict]:
        """Analyze detections for safety violations"""
        detection_results = []
        
        for person in detections['human']:
            analysis = self._analyze_person_safety(frame, person, detections)
            if analysis:
                detection_results.append(analysis)
        
        return detection_results
    
    def _analyze_person_safety(self, frame, person: Dict, detections: Dict) -> Optional[Dict]:
        """Analyze safety equipment for a single person"""
        regions = self._calculate_person_regions(person['bbox'])
        equipment_status = self._check_safety_equipment(regions, detections)
        risk_assessment = self._assess_risk(equipment_status)
        
        self._draw_detections(frame, person['bbox'], regions, equipment_status, risk_assessment)
        
        return self._create_detection_record(equipment_status, risk_assessment, frame)
    
    def _calculate_person_regions(self, bbox: Tuple[int, int, int, int]) -> Dict:
        """Calculate head and body regions for a person"""
        px1, py1, px2, py2 = bbox
        person_height = py2 - py1
        person_width = px2 - px1
        
        head_height = person_height // 3
        head_width = person_width // 1.5
        head_center_x = (px1 + px2) // 2
        
        return {
            'head': (
                int(head_center_x - head_width // 2),
                py1,
                int(head_center_x + head_width // 2),
                py1 + head_height
            ),
            'body': (px1, py1 + head_height, px2, py2)
        }
    
    def _check_safety_equipment(self, regions: Dict, detections: Dict) -> Dict:
        """Check if person is wearing required safety equipment"""
        return {
            'helmet': any(self._check_overlap(regions['head'], h['bbox']) 
                         for h in detections['hard_hat']),
            'vest': any(self._check_overlap(regions['body'], v['bbox']) 
                       for v in detections['safety_vest'])
        }
    
    @staticmethod
    def _check_overlap(region1: Tuple, region2: Tuple) -> bool:
        """Check if two regions overlap"""
        x1, y1, x2, y2 = region1
        x3, y3, x4, y4 = region2
        return not ((x1 >= x4) or (x2 <= x3) or (y1 >= y4) or (y2 <= y3))
    
    def _assess_risk(self, equipment_status: Dict) -> Dict:
        """Assess risk level based on missing equipment"""
        if not equipment_status['helmet'] and not equipment_status['vest']:
            return {'level': "MEDIUM", 'details': "전부 미착용"}
        elif not equipment_status['helmet']:
            return {'level': "LOW", 'details': "안전모 미착용"}
        elif not equipment_status['vest']:
            return {'level': "LOW", 'details': "안전조끼 미착용"}
        return {'level': "SAFE", 'details': "전부 착용"}
    
    def _create_detection_record(self, equipment_status: Dict, 
                               risk_assessment: Dict, frame) -> Optional[Dict]:
        """Create detection record if conditions are met"""
        current_time = datetime.now()
        
        if (current_time.timestamp() - self.last_capture_time >= 10 and 
            risk_assessment['level'] != "SAFE"):
            
            filename = f"captures/{current_time.strftime('%Y-%m-%d_%H_%M_%S')}_{risk_assessment['level']}.jpg"
            os.makedirs('captures', exist_ok=True)
            cv2.imwrite(filename, frame)
            
            self.last_capture_time = current_time.timestamp()
            
            return {
                "camera_id": "CAM_001",
                "detection_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "detection_object": equipment_status,
                "image_url": filename,
                "risk_level": risk_assessment['level'],
                "details": risk_assessment['details']
            }
        return None
    
    def _draw_detections(self, frame, bbox: Tuple, regions: Dict, 
                        equipment_status: Dict, risk_assessment: Dict):
        """Draw detection results on frame"""
        px1, py1, px2, py2 = bbox
        
        # Draw person bbox
        cv2.rectangle(frame, (px1, py1), (px2, py2), self.COLORS['human'], 2)
        
        # Draw equipment regions
        if equipment_status['helmet']:
            cv2.rectangle(frame, 
                         (regions['head'][0], regions['head'][1]),
                         (regions['head'][2], regions['head'][3]),
                         self.COLORS['hard_hat'], 2)
        if equipment_status['vest']:
            cv2.rectangle(frame, 
                         (regions['body'][0], regions['body'][1]),
                         (regions['body'][2], regions['body'][3]),
                         self.COLORS['safety_vest'], 2)
        
        # Draw status text
        text_color = (0, 255, 0) if all(equipment_status.values()) else (0, 0, 255)
        status_text = f"Safety Hat: {'OK' if equipment_status['helmet'] else 'X'}"
        status_text += f" | Vest: {'OK' if equipment_status['vest'] else 'X'}"
        
        cv2.putText(frame, status_text, (px1, py1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        cv2.putText(frame, f"Risk: {risk_assessment['level']}", 
                   (px1, py1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)