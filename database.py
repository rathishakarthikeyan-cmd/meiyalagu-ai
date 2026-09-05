import sqlite3
import os
from datetime import datetime

# Database file path (stored in the 'instance' folder)
DB_PATH = os.path.join('instance', 'history.db')

def get_db_connection():
    """Create a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Allows accessing columns by name
    return conn

def init_db():
    """
    Initialize the database.
    Creates the 'screenings' table if it doesn't exist.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS screenings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            image_path TEXT NOT NULL,
            gradcam_path TEXT,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            risk_level TEXT NOT NULL,
            recommendation TEXT NOT NULL
        )
    ''')

    conn.commit()
    conn.close()
    print("✅ Database initialized successfully!")

def save_screening(patient_name, image_path, gradcam_path, prediction, confidence, risk_level, recommendation):
    """
    Save a new screening record to the database.
    Returns: The ID of the newly inserted record.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute('''
        INSERT INTO screenings 
        (patient_name, timestamp, image_path, gradcam_path, prediction, confidence, risk_level, recommendation)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (patient_name, timestamp, image_path, gradcam_path, prediction, confidence, risk_level, recommendation))

    record_id = cursor.lastrowid
    conn.commit()
    conn.close()

    print(f"✅ Screening saved successfully! ID: {record_id}")
    return record_id

def get_all_screenings(limit=100):
    """
    Fetch all screenings, sorted by newest first.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT * FROM screenings 
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (limit,))

    records = cursor.fetchall()
    conn.close()
    return records

def get_last_screening_by_patient(patient_name):
    """
    Fetch the most recent screening for a specific patient.
    Used for trend detection (e.g., "Risk increased since last visit").
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT * FROM screenings 
        WHERE patient_name = ? 
        ORDER BY timestamp DESC 
        LIMIT 1
    ''', (patient_name,))

    record = cursor.fetchone()
    conn.close()
    return record

def get_screening_by_id(record_id):
    """
    Fetch a single screening by its ID.
    Useful for viewing detailed reports.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM screenings WHERE id = ?', (record_id,))
    record = cursor.fetchone()
    conn.close()
    return record