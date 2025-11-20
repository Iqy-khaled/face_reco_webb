import pyodbc
import os
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

class AccessDatabaseManager:
    def __init__(self, db_path):
        """Initialize connection to Microsoft Access database"""
        self.db_path = db_path
        
        # Verify database file exists
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Access database file not found: {db_path}")
            
        # Find appropriate Access driver
        self.driver = self._find_access_driver()
        
        # Ensure all required tables exist
        self._ensure_tables_exist()
        
    def _find_access_driver(self) -> str:
        """Find an appropriate MS Access ODBC driver"""
        drivers = [x for x in pyodbc.drivers() if 'Access' in x]
        if not drivers:
            raise EnvironmentError(
                "No Microsoft Access ODBC driver found. Please install Microsoft Access Database Engine."
            )
        # Prefer newer Access drivers (for .accdb) over older ones (for .mdb)
        for preferred in ['Microsoft Access Driver (*.accdb)', 'Microsoft Access Driver (*.accdb, *.mdb)']:
            if preferred in drivers:
                return preferred
        return drivers[0]
    
    def _get_connection_string(self) -> str:
        """Generate the ODBC connection string"""
        return f'DRIVER={{{self.driver}}};DBQ={os.path.abspath(self.db_path)};'
    
    @contextmanager
    def get_connection(self):
        """Get a database connection with context management"""
        conn = None
        try:
            conn = pyodbc.connect(self._get_connection_string(), autocommit=False)
            yield conn
        except pyodbc.Error as e:
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    
    def _ensure_tables_exist(self):
        """Create required tables if they don't exist"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if people table exists, create if not
            if not self._table_exists(cursor, 'people'):
                cursor.execute('''
                CREATE TABLE people (
                    person_id COUNTER PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    email VARCHAR(255),
                    department VARCHAR(255),
                    role VARCHAR(255)
                )
                ''')
                
            # Check if face_encodings table exists, create if not
            if not self._table_exists(cursor, 'face_encodings'):
                cursor.execute('''
                CREATE TABLE face_encodings (
                    encoding_id COUNTER PRIMARY KEY,
                    person_id INTEGER NOT NULL,
                    encoding_data LONGBINARY NOT NULL,
                    image_data LONGBINARY
                )
                ''')
                
            # Check if attendance table exists, create if not
            if not self._table_exists(cursor, 'attendance'):
                cursor.execute('''
                CREATE TABLE attendance (
                    attendance_id COUNTER PRIMARY KEY,
                    person_id INTEGER NOT NULL,
                    check_in_time DATETIME NOT NULL,
                    check_out_time DATETIME,
                    confidence_score DOUBLE
                )
                ''')
                
            conn.commit()
    
    def _table_exists(self, cursor, table_name):
        """Check if a table exists in the database"""
        try:
            cursor.tables(table=table_name).fetchone()
            return True
        except:
            return False
    
    def fetchall(self, query, params=()):
        """Execute a query and return all results as a list of dictionaries"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            if cursor.description:
                columns = [column[0] for column in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
            return []
    
    def fetchone(self, query, params=()):
        """Execute a query and return the first result as a dictionary"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            if cursor.description:
                row = cursor.fetchone()
                if row:
                    columns = [column[0] for column in cursor.description]
                    return dict(zip(columns, row))
            return None
            
    def execute(self, query, params=()):
        """Execute a query and return the cursor"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor

# For backwards compatibility, keep the original name
DatabaseManager = AccessDatabaseManager
