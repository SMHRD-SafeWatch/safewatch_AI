import os
from dotenv import load_dotenv

class DatabaseConfig:
    def __init__(self):
        load_dotenv()
        
        self.user = os.getenv('DB_USER')
        self.password = os.getenv('DB_PASSWORD')
        self.host = os.getenv('DB_HOST', '')
        self.port = os.getenv('DB_PORT', '')
        self.sid = os.getenv('DB_SID', '')
        
    def get_dsn(self):
        return f"{self.host}:{self.port}/{self.sid}"
    
    def get_connection_params(self):
        return {
            'user' : self.user,
            'password' : self.password,
            'dsn' : self.get_dsn()
        }