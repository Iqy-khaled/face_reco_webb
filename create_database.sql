CREATE DATABASE IF NOT EXISTS attendance_system;

USE attendance_system;

CREATE TABLE IF NOT EXISTS face_files (
    id INT AUTO_INCREMENT PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Face Encodings table - store facial embeddings and original images
CREATE TABLE face_encodings (
    encoding_id INT AUTO_INCREMENT PRIMARY KEY,
    person_id INT NOT NULL,
    encoding_data BLOB NOT NULL, -- Stores the numpy array bytes
    image_data BLOB,  -- Optional: Store the original registration image
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (person_id) REFERENCES people(person_id) ON DELETE CASCADE
);
