import cv2
from datetime import datetime

def image_encode(display_frame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 이미지를 메모리에서 바로 jpg 형식으로 인코딩
    success, encoded_image = cv2.imencode('.jpg', display_frame)
    if success:
        # 인코딩된 이미지를 바이너리 데이터로 변환
        image_binary = encoded_image.tobytes()
        print('이미지 인코딩 완료')
        return timestamp, image_binary
    else :
        print('이미지 인코딩 실패')