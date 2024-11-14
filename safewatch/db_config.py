import cx_Oracle
from datetime import datetime
import json
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from database.config import DatabaseConfig

class OracleDB:
    def __init__(self):
        db_config = DatabaseConfig()
        
        # 접속 정보 가져오기
        connection_params = db_config.get_connection_params()
        
        # Oracle 접속
        self.connection = cx_Oracle.connect(**connection_params)
        
    def insert_detection(self, camera_id: str, detection_time: datetime, 
                        detection_object: dict, risk_level: str, 
                        content: str, image_url: bytes):
        cursor = self.connection.cursor()
        try:
            detection_object_json = json.dumps(detection_object)
            
            blob_var = cursor.var(cx_Oracle.BLOB)
            blob_var.setvalue(0, image_url)
            
            cursor.execute("""
                INSERT INTO detection
                (detection_id, camera_id, detection_time, detection_object, 
                 risk_level, content, image_url)
                VALUES 
                (detection_id_seq.NEXTVAL, :1, :2, :3, :4, :5, :6)
                """, 
                (camera_id, detection_time, detection_object_json,
                 risk_level, content, blob_var)
            )
            
            self.connection.commit()
            
        except Exception as e:
            print(f"Error inserting data: {e}")
            self.connection.rollback()
            raise
        finally:
            cursor.close()
            
    def close(self):
        if self.connection:
            self.connection.close()



