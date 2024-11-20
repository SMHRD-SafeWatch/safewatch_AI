# database.py
import os
import cx_Oracle
from dotenv import load_dotenv

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

def get_connection():
    """데이터베이스 연결을 반환합니다."""
    try:
        connection = cx_Oracle.connect(user=username, password=password, dsn=dsn)
        print("Database connected successfully.")
        return connection
    except cx_Oracle.DatabaseError as e:
        print(f"Failed to connect to the database: {e}")
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
        print("Data inserted successfully.")
    
    except cx_Oracle.DatabaseError as e:
        print(f"Database error occurred: {e}")
        if connection:
            connection.rollback()
        raise
    
    finally:
        # 리소스 해제
        if cursor:
            cursor.close()
        if connection:
            connection.close()
