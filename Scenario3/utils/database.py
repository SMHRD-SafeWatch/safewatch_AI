import cx_Oracle
from utils import config
from dotenv import load_dotenv
import threading
import os

# .env 파일 로드(환경변수 보안관련 내용)
load_dotenv()

# 데이터베이스 연결 정보 로드
username = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
sid = os.getenv("DB_SID")

# DSN(Data Source Name) 생성
dsn = cx_Oracle.makedsn(host, port, sid=id)

# DB 연결
def get_connection():
    """데이터베이스 연결을 반환합니다."""
    return cx_Oracle.connect(user=username, password=password, dsn=dsn)


#DB insert
def insert_detection_data(camera_id, detection_time, detection_object, image_url, risk_level, content):
    connection = None
    try:
        # 데이터베이스 연결
        connection = get_connection()
        cursor = connection.cursor()

        # SQL 삽입 쿼리
        insert_query = """
        INSERT INTO DETECTION (DETECTION_ID, CAMERA_ID, DETECTION_TIME, DETECTION_OBJECT, IMAGE_URL, RISK_LEVEL, CONTENT)
        VALUES (detection_id_seq.NEXTVAL, :camera_id, :detection_time, :detection_object, :image_url, :risk_level, :content)
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
        print("Data inserted successfully.")

    except cx_Oracle.DatabaseError as e:
        print(f"Database error occurred: {e}")
        if connection:
            connection.rollback()

    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()
            
#async 스레딩 사용하여 병렬로 처리
def async_insert_detection_data(*args):
    threading.Thread(target=insert_detection_data, args=args).start()