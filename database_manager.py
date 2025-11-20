# <llm-snippet-file>app/services/database.py</llm-snippet-file>
import sqlite3
import os
from contextlib import contextmanager
from typing import Optional, List, Dict, Any

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_db_directory()

    def _ensure_db_directory(self):
        """Ensures the directory for the database file exists."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            print(f"Created database directory: {db_dir}")

    @contextmanager
    def get_connection(self):
        """Provides a transactional scope around a series of operations."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Return rows as dictionary-like objects
            yield conn
            conn.commit()
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            if conn:
                conn.rollback()
            raise  # Re-raise the exception after rollback
        finally:
            if conn:
                conn.close()

    def execute(self, query: str, params: tuple = ()):
        """Executes a single SQL statement."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor

    def fetchall(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Executes a query and returns all results."""
        with self.get_connection() as conn:
             cursor = conn.cursor()
             cursor.execute(query, params)
             # Convert sqlite3.Row objects to plain dictionaries
             return [dict(row) for row in cursor.fetchall()]


    def fetchone(self, query: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        """Executes a query and returns a single result."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()
            return dict(row) if row else None


    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_db_directory()
        self.init_db()  # Call init_db here
    
    def _ensure_db_directory(self):
        """Ensures the directory for the database file exists."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            print(f"Created database directory: {db_dir}")
    
    def init_db(self):
        """Initializes the database schema if tables don't exist."""
        # Updated people table to include 'role'
        schema = """
        CREATE TABLE IF NOT EXISTS people (
            person_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            email TEXT,
            department TEXT,
            role TEXT,                     -- Added role/position field
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS face_encodings (
            encoding_id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            encoding_data BLOB NOT NULL, -- Stores the numpy array bytes
            image_data BLOB,             -- Optional: Store the original registration image
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (person_id) REFERENCES people (person_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS attendance (
            attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            check_in_time DATETIME NOT NULL,
            check_out_time DATETIME,
            confidence_score REAL,      -- Store recognition confidence
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (person_id) REFERENCES people (person_id) ON DELETE CASCADE
        );
        """
        # Add indexes for performance
        indexes = """
        CREATE INDEX IF NOT EXISTS idx_attendance_person_time ON attendance (person_id, check_in_time);
        CREATE INDEX IF NOT EXISTS idx_people_name ON people (name);
        -- Optional: Index for department or role if frequently queried
        CREATE INDEX IF NOT EXISTS idx_people_department ON people (department);
        CREATE INDEX IF NOT EXISTS idx_people_role ON people (role);
        """
        try:
            with self.get_connection() as conn:
                conn.executescript(schema)
                conn.executescript(indexes)
            print("Database schema checked/initialized successfully.")
        except sqlite3.Error as e:
            print(f"Error initializing database schema: {e}")
            raise