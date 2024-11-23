# database.py
import os
import cx_Oracle
from dotenv import load_dotenv
import threading

# .env 파일 로드
load_dotenv()

# 데이터베이스 연결 정보 로드
username = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
sid = os.getenv("DB_SID")

# DSN(Data Source Name) 생성
dsn = cx_Oracle.makedsn(host, port, sid=sid)

# 데이터베이스 삽입 기능 on/off 플래그 : 테스트 모드에서는 False로 설정하여 데이터베이스 삽입을 비활성화
DB_INSERT_ENABLED = False  # True: 데이터 삽입 활성화, False: 비활성화

def set_db_insert_enabled(enabled):
    """데이터 삽입 기능 활성화/비활성화 설정"""
    global DB_INSERT_ENABLED
    DB_INSERT_ENABLED = enabled
    print(f"DB Insert Enabled: {DB_INSERT_ENABLED}")

def get_connection():
    """데이터베이스 연결을 반환합니다."""
    try:
        connection = cx_Oracle.connect(user=username, password=password, dsn=dsn)
        print("DB 연결 성공")
        return connection
    except cx_Oracle.DatabaseError as e:
        print(f"DB 연결 실패: {e}")
        raise

def insert_detection_data(camera_id, detection_time, detection_object,
                          image_url, risk_level, content):
    """위험 탐지 데이터를 테이블에 삽입합니다."""
    connection = None
    cursor = None
    try:
        # 데이터베이스 연결
        connection = get_connection()
        cursor = connection.cursor()
        
        # SQL 삽입 쿼리
        insert_query = """
        INSERT INTO DETECTION (DETECTION_ID, CAMERA_ID, DETECTION_TIME,
        DETECTION_OBJECT, IMAGE_URL, RISK_LEVEL, CONTENT)
        VALUES (detection_id_seq.NEXTVAL, :camera_id, TO_DATE(:detection_time, 'YYYY-MM-DD HH24:MI:SS'),
        :detection_object, :image_url, :risk_level, :content)
        """
        
        # 데이터 삽입
        cursor.execute(insert_query, {
            "camera_id": camera_id,
            "detection_time": detection_time,
            "detection_object": detection_object,
            "image_url": image_url,
            "risk_level": risk_level,
            "content": content
        })
        
        # 변경 사항 커밋
        connection.commit()
        print("데이터 입력 성공.")
    
    except cx_Oracle.DatabaseError as e:
        print(f"DB 에러 발생: {e}")
        if connection:
            connection.rollback()
        raise
    
    finally:
        # 리소스 해제
        if cursor:
            cursor.close()
        if connection:
            connection.close()

#async 스레딩 사용하여 병렬로 처리
def async_insert_detection_data(*args):
    threading.Thread(target=insert_detection_data, args=args).start()