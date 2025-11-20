from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import tensorflow as tf
# noinspection PyUnresolvedReferences
from tensorflow.keras.applications import EfficientNetB0 # type: ignore
from tensorflow.keras.layers import GlobalAveragePooling2D # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
import json
import base64
import logging
from dataclasses import dataclass
from typing import Optional
from PIL import Image
import io
from datetime import datetime, timedelta
from database_manager import DatabaseManager  # Import from database_manager.py


# DatabaseManager is now imported from database_manager.py


@dataclass
class AppConfig:
    db_path: str
    secret_key: str
    model_name: str
    recognition_threshold: float
    image_size: tuple = (224, 224)

    @classmethod
    def from_file(cls, config_path: str) -> 'AppConfig':
        if not os.path.exists(config_path):
            return cls.get_default_config()

        with open(config_path, 'r') as f:
            config = json.load(f)

        return cls(
            db_path=config.get('database', {}).get('path', 'face_recognition_attendance.db'),
            secret_key=config.get('app', {}).get('secret_key', os.urandom(24).hex()),
            model_name=config.get('facial_recognition', {}).get('model_name', 'efficientnet'),
            recognition_threshold=config.get('facial_recognition', {}).get('threshold', 0.6)
        )

    @staticmethod
    def get_default_config() -> 'AppConfig':
        return AppConfig(
            db_path='face_recognition_attendance.db',
            secret_key=os.urandom(24).hex(),
            model_name='efficientnet',
            recognition_threshold=0.6
        )


def process_image(image_bytes: bytes) -> tf.Tensor:
    # Load and preprocess image
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((224, 224))
    image_array = img_to_array(image)
    image_array = tf.keras.applications.efficientnet.preprocess_input(image_array)
    return tf.convert_to_tensor([image_array])
# app.py
import os
# Set TensorFlow environment variable
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


logging.basicConfig(level=logging.DEBUG)

@dataclass
class AppConfig:
    db_path: str
    secret_key: str
    model_name: str
    recognition_threshold: float
    image_size: tuple = (224, 224)

    @classmethod
    def from_file(cls, config_path: str) -> 'AppConfig':
        if not os.path.exists(config_path):
            return cls.get_default_config()
        with open(config_path, 'r') as f:
            config = json.load(f)
        return cls(
            db_path=config.get('database', {}).get('path', 'face_recognition_attendance.db'),
            secret_key=config.get('app', {}).get('secret_key', os.urandom(24).hex()),
            model_name=config.get('facial_recognition', {}).get('model_name', 'efficientnet'),
            recognition_threshold=config.get('facial_recognition', {}).get('threshold', 0.6)
        )

    @staticmethod
    def get_default_config() -> 'AppConfig':
        return AppConfig(
            db_path='face_recognition_attendance.db',
            secret_key=os.urandom(24).hex(),
            model_name='efficientnet',
            recognition_threshold=0.6
        )

def process_image(image_bytes: bytes) -> tf.Tensor:
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((224, 224))
    image_array = img_to_array(image)
    image_array = tf.keras.applications.efficientnet.preprocess_input(image_array)
    return tf.convert_to_tensor([image_array])

def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    if embedding1 is None or embedding2 is None:
        return 0.0
    embedding1_norm = embedding1 / np.linalg.norm(embedding1)
    embedding2_norm = embedding2 / np.linalg.norm(embedding2)
    return float(np.dot(embedding1_norm, embedding2_norm))

class FaceRecognitionModel:
    def __init__(self):
        try:
            base_model = EfficientNetB0(include_top=False, weights='imagenet')
            self.model = tf.keras.Sequential([
                base_model,
                GlobalAveragePooling2D()
            ])
        except Exception as e:
            raise RuntimeError(f"Failed to initialize EfficientNetB0: {str(e)}")

    def extract_embedding(self, image_bytes: bytes) -> np.ndarray:
        preprocessed_image = process_image(image_bytes)
        embedding = self.model.predict(preprocessed_image)
        return embedding[0]

    @staticmethod
    def tensor_to_blob(embedding: np.ndarray) -> Optional[bytes]:
        if embedding is None:
            return None
        return embedding.astype(np.float32).tobytes()

    @staticmethod
    def blob_to_tensor(blob_data: Optional[bytes]) -> Optional[np.ndarray]:
        if blob_data is None:
            return None
        return np.frombuffer(blob_data, dtype=np.float32)

class AttendanceSystem:
    def __init__(self, config: AppConfig):
        self.config = config
        self.db = DatabaseManager(config.db_path)
        self.face_model = FaceRecognitionModel()
        self.app = self._create_app()

    def _create_app(self) -> Flask:
        app = Flask(__name__, static_folder='static')
        app.secret_key = self.config.secret_key
        self._register_routes(app)
        return app

    def recognize_face(self, image_bytes: bytes, threshold: float = 0.6) -> Optional[dict]:
        try:
            query_embedding = self.face_model.extract_embedding(image_bytes)
            stored_faces = self.db.fetchall('''
                SELECT fe.person_id, fe.encoding_data, p.name, p.role
                FROM face_encodings fe 
                JOIN people p ON fe.person_id = p.person_id
            ''')
            best_match = None
            highest_similarity = -1
            for face in stored_faces:
                stored_embedding = self.face_model.blob_to_tensor(face['encoding_data'])
                similarity = calculate_similarity(query_embedding, stored_embedding)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = {
                        'person_id': face['person_id'],
                        'name': face['name'],
                        'pos': face['role'] if face['role'] else 'No Position',
                        'confidence': similarity
                    }
            if best_match and best_match['confidence'] >= threshold:
                return best_match
            return None
        except Exception as e:
            logging.error(f"Face recognition error: {str(e)}")
            return None

    def _register_routes(self, app):
        @app.route('/')
        def index():
            return render_template("index.html")

        @app.route('/api/register', methods=['POST'])
        def register():
            data = request.json
            name = data.get('name')
            image_data = data.get('image')
            if not name or not image_data:
                return jsonify({'error': 'Name and image required'}), 400
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            try:
                image_bytes = base64.b64decode(image_data)
                embedding = self.face_model.extract_embedding(image_bytes)
                
                # With SQLite, we can get the last row id directly
                with self.db.get_connection() as conn:
                    cursor = conn.execute('INSERT INTO people (name) VALUES (?)', (name,))
                    person_id = cursor.lastrowid
                    
                    conn.execute(
                        'INSERT INTO face_encodings (person_id, encoding_data, image_data) VALUES (?, ?, ?)',
                        (person_id, self.face_model.tensor_to_blob(embedding), image_bytes)
                    )
                    conn.commit()
                    
                return jsonify({'success': True, 'person_id': person_id})
            except Exception as e:
                logging.error(f"Register error: {str(e)}")
                return jsonify({'error': str(e)}), 500

        @app.route('/api/people', methods=['GET'])
        def get_people():
            try:
                people = self.db.fetchall('SELECT person_id, name, email, department, role FROM people')
                return jsonify(people)
            except Exception as e:
                logging.error(f"Get people error: {str(e)}")
                return jsonify({'error': str(e)}), 500

        @app.route('/api/face-image/<int:person_id>', methods=['GET'])
        def get_face_image(person_id):
            try:
                face_data = self.db.fetchone(
                    'SELECT image_data FROM face_encodings WHERE person_id = ? LIMIT 1',
                    (person_id,)
                )
                if not face_data or not face_data['image_data']:
                    return jsonify({'error': 'No image found for this person'}), 404
                image_base64 = base64.b64encode(face_data['image_data']).decode('utf-8')
                return jsonify({'person_id': person_id, 'image': f'data:image/jpeg;base64,{image_base64}'})
            except Exception as e:
                logging.error(f"Get face image error: {str(e)}")
                return jsonify({'error': str(e)}), 500

        @app.route('/api/recognize', methods=['POST'])
        def recognize():
            data = request.json
            image_data = data.get('image')
            is_check_out = data.get('is_check_out', False)
            if not image_data:
                return jsonify({'error': 'Image required'}), 400
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            try:
                image_bytes = base64.b64decode(image_data)
                result = self.recognize_face(image_bytes, self.config.recognition_threshold)
                if not result:
                    return jsonify({'error': 'No matching face found'}), 404
                with self.db.get_connection() as conn:
                    check_in_time = datetime.now()
                    cursor = conn.execute(
                        'INSERT INTO attendance (person_id, check_in_time, check_out_time, confidence_score) VALUES (?, ?, ?, ?)',
                        (result['person_id'], check_in_time, check_in_time if is_check_out else None, result['confidence'])
                    )
                    attendance_id = cursor.lastrowid
                    conn.commit()
                result.update({
                    'attendance_id': attendance_id,
                    'timestamp': check_in_time.isoformat(),
                    'is_check_out': is_check_out
                })
                return jsonify(result)
            except Exception as e:
                logging.error(f"Recognize error: {str(e)}")
                return jsonify({'error': str(e)}), 500

        @app.route('/api/attendance/today', methods=['GET'])
        def get_today_attendance():
            try:
                # Get today's date bounds for SQLite
                today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                today_end = today_start + timedelta(days=1)
                
                # For SQLite, convert to ISO format string
                today_start_str = today_start.isoformat()
                today_end_str = today_end.isoformat()
                
                records = self.db.fetchall('''
                    SELECT a.attendance_id, a.person_id, p.name, a.check_in_time, a.check_out_time, a.confidence_score
                    FROM attendance a
                    JOIN people p ON a.person_id = p.person_id
                    WHERE a.check_in_time >= ? AND a.check_in_time < ?
                    ORDER BY a.check_in_time DESC
                ''', (today_start_str, today_end_str))
                
                # Format datetime objects to string for JSON serialization if needed
                # Note: SQLite might already return these as strings
                for record in records:
                    if 'check_in_time' in record and record['check_in_time'] and not isinstance(record['check_in_time'], str):
                        record['check_in_time'] = record['check_in_time'].isoformat()
                    if 'check_out_time' in record and record['check_out_time'] and not isinstance(record['check_out_time'], str):
                        record['check_out_time'] = record['check_out_time'].isoformat()
                        
                return jsonify(records)
            except Exception as e:
                logging.error(f"Get attendance error: {str(e)}")
                return jsonify({'error': str(e)}), 500

def create_app() -> Flask:
    config = AppConfig.from_file('config.json')
    system = AttendanceSystem(config)
    return system.app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)

def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings"""
    if embedding1 is None or embedding2 is None:
        return 0.0

    # Normalize embeddings
    embedding1_norm = embedding1 / np.linalg.norm(embedding1)
    embedding2_norm = embedding2 / np.linalg.norm(embedding2)

    # Calculate cosine similarity
    similarity = np.dot(embedding1_norm, embedding2_norm)
    return float(similarity)


class FaceRecognitionModel:
    def __init__(self):
        # Initialize EfficientNet model without top layers
        base_model = EfficientNetB0(include_top=False, weights='imagenet')
        self.model = tf.keras.Sequential([
            base_model,
            GlobalAveragePooling2D()
        ])

    def extract_embedding(self, image_bytes: bytes) -> np.ndarray:
        """Extract face embedding using EfficientNet"""
        preprocessed_image = process_image(image_bytes)
        embedding = self.model.predict(preprocessed_image)
        return embedding[0]  # Return the first (and only) embedding

    @staticmethod
    def tensor_to_blob(embedding: np.ndarray) -> Optional[bytes]:
        """Convert numpy array to blob for database storage"""
        if embedding is None:
            return None
        return embedding.astype(np.float32).tobytes()

    @staticmethod
    def blob_to_tensor(blob_data: Optional[bytes]) -> Optional[np.ndarray]:
        """Convert blob to numpy array"""
        if blob_data is None:
            return None
        return np.frombuffer(blob_data, dtype=np.float32)


from datetime import datetime, timedelta

class AttendanceSystem:
    def __init__(self, config: AppConfig):
        self.config = config
        self.db = DatabaseManager(config.db_path)
        self.face_model = FaceRecognitionModel()
        self.app = self._create_app()
        print(f"Initialized AttendanceSystem with config: {config}")

    def _create_app(self) -> Flask:
        app = Flask(__name__, static_folder='static')
        app.secret_key = self.config.secret_key
        self._register_routes(app)
        return app

    def _process_image_data(self, image_data):
        """Helper method to process base64 image data"""
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        return base64.b64decode(image_data)

    def _handle_registration(self, data):
        """Handle user registration logic"""
        name = data.get('name')
        image_data = data.get('image')

        if not name or not image_data:
            return jsonify({'error': 'Name and image required'}), 400

        try:
            image_bytes = self._process_image_data(image_data)
            embedding = self.face_model.extract_embedding(image_bytes)

            with self.db.get_connection() as conn:
                cursor = conn.execute('INSERT INTO people (name) VALUES (?)', (name,))
                person_id = cursor.lastrowid
                conn.execute(
                    'INSERT INTO face_encodings (person_id, encoding_data, image_data) VALUES (?, ?, ?)',
                    (person_id, self.face_model.tensor_to_blob(embedding), image_bytes)
                )
                conn.commit()

            return jsonify({'success': True, 'person_id': person_id})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def _handle_recognition(self, data):
        """Handle face recognition logic"""
        image_data = data.get('image')
        is_check_out = data.get('is_check_out', False)

        if not image_data:
            return jsonify({'error': 'Image required'}), 400

        try:
            image_bytes = self._process_image_data(image_data)
            result = self.recognize_face(image_bytes, self.config.recognition_threshold)

            if not result:
                return jsonify({'error': 'No matching face found'}), 404

            # Record attendance - use ISO format string for SQLite dates
            check_in_time = datetime.now()
            check_in_time_str = check_in_time.isoformat()
            check_out_time_str = check_in_time.isoformat() if is_check_out else None
            
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    'INSERT INTO attendance (person_id, check_in_time, check_out_time, confidence_score) VALUES (?, ?, ?, ?)',
                    (result['person_id'], check_in_time_str, check_out_time_str, result['confidence'])
                )
                attendance_id = cursor.lastrowid
                conn.commit()
    
            result.update({
                'attendance_id': attendance_id,
                'timestamp': check_in_time.isoformat(),
                'is_check_out': is_check_out
            })
            return jsonify(result)
        except Exception as e:
            logging.error(f"Recognition error: {str(e)}")
            return jsonify({'error': str(e)}), 500

    def _get_people_list(self):
        """Get list of registered people"""
        try:
            people = self.db.fetchall('SELECT person_id, name, email, department, role FROM people')
            return jsonify(people)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def _get_face_image(self, person_id):
        """Get face image for a specific person"""
        try:
            face_data = self.db.fetchone(
                'SELECT image_data FROM face_encodings WHERE person_id = ? LIMIT 1',
                (person_id,)
            )
            if not face_data or not face_data['image_data']:
                return jsonify({'error': 'No image found for this person'}), 404
            image_base64 = base64.b64encode(face_data['image_data']).decode('utf-8')
            return jsonify({'person_id': person_id, 'image': f'data:image/jpeg;base64,{image_base64}'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def _register_routes(self, app):
        import base64
        from datetime import datetime

        @app.route('/')
        def index():
            return app.send_static_file('index.html')

        @app.route('/api/register', methods=['POST'])
        def register():
            return self._handle_registration(request.json)

        @app.route('/api/recognize', methods=['POST'])
        def recognize():
            return self._handle_recognition(request.json)
    
        @app.route('/api/people', methods=['GET'])
        def get_people():
            return self._get_people_list()
    
        @app.route('/api/face-image/<int:person_id>', methods=['GET'])
        def get_face_image(person_id):
            return self._get_face_image(person_id)
    
        @app.route('/api/image-sizes', methods=['GET'])
        def get_image_sizes():
            try:
                images = self.db.fetchall("""
                    SELECT fe.person_id, p.name, LENGTH(fe.image_data) as image_size
                    FROM face_encodings fe
                    JOIN people p ON fe.person_id = p.person_id
                    WHERE fe.image_data IS NOT NULL
                """)
                total_images = len(images)
                total_size = sum(img['image_size'] for img in images)
                avg_size = total_size / total_images if total_images > 0 else 0
    
                summary = {
                    'total_images': total_images,
                    'total_size_mb': total_size / (1024 * 1024),
                    'avg_size_kb': avg_size / 1024
                }
                return jsonify({'images': images, 'summary': summary})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
        @app.route('/api/import-known-faces', methods=['POST'])
        def import_known_faces():
            import glob
            known_faces_dir = 'known_faces'
            if not os.path.exists(known_faces_dir):
                return jsonify({'message': 'Known faces directory not found', 'imported': 0, 'skipped': 0, 'error': 0, 'details': []}), 400
    
            results = {'imported': 0, 'skipped': 0, 'error': 0, 'details': []}
            image_files = glob.glob(os.path.join(known_faces_dir, '*.jpg')) + glob.glob(os.path.join(known_faces_dir, '*.png'))
    
            for file_path in image_files:
                filename = os.path.basename(file_path)
                name = os.path.splitext(filename)[0].replace('_', ' ')
                try:
                    with open(file_path, 'rb') as f:
                        image_bytes = f.read()
                    embedding = self.face_model.extract_embedding(image_bytes)
    
                    # Check if person already exists
                    existing = self.db.fetchone('SELECT person_id FROM people WHERE name = ?', (name,))
                    if existing:
                        results['skipped'] += 1
                        results['details'].append({'file': filename, 'status': 'skipped', 'reason': 'Person already exists'})
                        continue
            
                    # Insert new person
                    with self.db.get_connection() as conn:
                        cursor = conn.execute('INSERT INTO people (name) VALUES (?)', (name,))
                        person_id = cursor.lastrowid
                        conn.execute(
                            'INSERT INTO face_encodings (person_id, encoding_data, image_data) VALUES (?, ?, ?)',
                            (person_id, self.face_model.tensor_to_blob(embedding), image_bytes)
                        )
                        conn.commit()
            
                    results['imported'] += 1
                    results['details'].append({'file': filename, 'status': 'imported', 'person_id': person_id, 'name': name})
                except Exception as e:
                    logging.error(f"Import error for {filename}: {str(e)}")
                    results['error'] += 1
                    results['details'].append({'file': filename, 'status': 'error', 'reason': str(e)})
    
            message = f'Import completed: {results["imported"]} imported, {results["skipped"]} skipped, {results["error"]} errors'
            return jsonify({'message': message, **results})
    
        @app.route('/api/attendance/today', methods=['GET'])
        def get_today_attendance():
            try:
                from datetime import datetime, timedelta
                today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                tomorrow = today + timedelta(days=1)
    
                records = self.db.fetchall("""
                    SELECT a.attendance_id, a.person_id, p.name, a.check_in_time, a.check_out_time, a.confidence_score
                    FROM attendance a
                    JOIN people p ON a.person_id = p.person_id
                    WHERE a.check_in_time >= ? AND a.check_in_time < ?
                """, (today.isoformat(), tomorrow.isoformat()))
                return jsonify(records)
            except Exception as e:
                return jsonify({'error': str(e)}), 500

    def recognize_face(self, image_bytes: bytes, threshold: float = 0.6) -> Optional[dict]:
        """Recognize face in image and return matching person if found"""
        try:
            # Extract embedding for the input image
            query_embedding = self.face_model.extract_embedding(image_bytes)

            # Get stored embeddings from database
            stored_faces = self.db.fetchall('''
                SELECT fe.person_id, fe.encoding_data, p.name, p.role 
                FROM face_encodings fe 
                JOIN people p ON fe.person_id = p.person_id
            ''')
    
            best_match = None
            highest_similarity = -1
    
            for face in stored_faces:
                stored_embedding = self.face_model.blob_to_tensor(face['encoding_data'])
                similarity = calculate_similarity(query_embedding, stored_embedding)
    
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = {
                        'person_id': face['person_id'],
                        'name': face['name'],
                        'pos': face['role'] if face['role'] else 'No Position',
                        'confidence': similarity
                    }
    
            if best_match and best_match['confidence'] >= threshold:
                return best_match
            return None
    
        except Exception as e:
            logging.error(f"Face recognition error: {str(e)}")
            return None


def create_app() -> Flask:
    config = AppConfig.from_file('config.json')
    system = AttendanceSystem(config)
    return system.app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
