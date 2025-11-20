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