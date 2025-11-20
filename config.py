import os

class AppConfig:
    def __init__(self):
        # Use the Access database file
        self.db_path = os.path.abspath('face_recognition_attendance.accdb')
        self.secret_key = 'your-secret-key-here'
        self.recognition_threshold = 0.6
        
    def __str__(self):
        return f"AppConfig(db_path={self.db_path}, recognition_threshold={self.recognition_threshold})"
