import cv2
from camera import get_camera
import detect
from utils import zone

cap = get_camera()
if cap is None:
    exit()
    
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("캠에서 이미지를 가져올 수 없습니다.")
        break
    
    #프레임 크기를 조정 (예: 1280x720으로 조정하여 노트북 화면에 맞춤)
    frame = cv2.resize(frame, (1535,820))  # 원하는 너비와 높이로 조절 
    display_frame = frame.copy()
    
    #구역표시
    displayed_frame = zone.annotate_zones(display_frame)
    
    #프레임에서 손 감지
    results= detect.detect_objects(frame)
    detect.hand_detections(results, display_frame)
    
    cv2.imshow('SAFEWATCH - Hand Detection', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
