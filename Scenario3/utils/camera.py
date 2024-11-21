import cv2
from utils import config

rtsp_url = f"rtsp://{config.ipcam_username}:{config.ipcam_password}@{config.ip_address}{config.stream_path}"

def get_camera():
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("웹캠에서 이미지를 가져올 수 없습니다.")
        cap.release()
        return None
    return cap