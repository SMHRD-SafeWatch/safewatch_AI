import cv2
from dotenv import load_dotenv
import os

# .env 파일 로드(환경변수 보안관련 내용)
load_dotenv()

# ipcam 연결 정보 로드
ipcam_username = os.getenv("ipcam_username")
ipcam_password = os.getenv("ipcam_password")
ip_address = os.getenv("ip_address")
stream_path = os.getenv("stream_path")

rtsp_url = f"rtsp://{ipcam_username}:{ipcam_password}@{ip_address}{stream_path}"

def get_camera():
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("웹캠에서 이미지를 가져올 수 없습니다.")
        cap.release()
        return None
    return cap