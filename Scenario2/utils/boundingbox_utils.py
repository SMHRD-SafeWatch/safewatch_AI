# utils/boundingbox_utils.py
import cv2

def draw_bounding_boxes(frame, detections, colors):
    """탐지된 객체들의 바운딩 박스를 프레임에 그립니다."""
    for obj_class, objects in detections.items():
        color = colors.get(obj_class, (255, 255, 255))  # 클래스별 색상 설정
        for obj in objects:
            x1, y1, x2, y2 = obj['bbox']
            conf = obj['conf']
            # 바운딩 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # 클래스 이름과 신뢰도 표시
            label = f"{obj_class} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def draw_status(frame, status_texts):
    """프레임에 상태 정보를 표시합니다."""
    y_offset = 20
    for i, (text, color) in enumerate(status_texts):
        cv2.putText(frame, text, (10, y_offset + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
