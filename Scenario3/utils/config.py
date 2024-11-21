### config.py ###

# 카메라 설정
ipcam_username = "safewatch"  # 사용자명 입력
ipcam_password = "123456"  # 비밀번호 입력
ip_address = "192.168.20.213"  # IP 주소 입력
stream_path = "/stream1"  # 스트림 경로

# 데이터베이스 설정
db_username = "Insa5_SpringB_final_3"
db_password = 'aischool3'
host = 'project-db-stu3.smhrd.com'
port = 1524
sid = 'xe'

# 위험구역 범위 설정 
warning_zone_start, warning_zone_end = (215, 90), (500, 660)
danger_zone_start, danger_zone_end = (80,150), (350, 600)
