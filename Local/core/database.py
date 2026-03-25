import sqlite3
import os
import json

# Support for persistent storage on Render
DATA_DIR = os.getenv("PERSISTENT_STORAGE_PATH", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(DATA_DIR, "chat_history.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, timeout=20)
    conn.row_factory = sqlite3.Row
    # Enable WAL mode for better concurrent access in Gunicorn
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def init_db():
    conn = get_db_connection()
    try:
        with conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS pending_confirmations (
                    sender_jid TEXT PRIMARY KEY,
                    message_id TEXT,
                    output_path TEXT,
                    original_filename TEXT,
                    state TEXT,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS user_status (
                    sender_jid TEXT PRIMARY KEY,
                    status TEXT DEFAULT 'bot',
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS validated_registries (
                    unique_id TEXT PRIMARY KEY,
                    sender_jid TEXT,
                    cedula TEXT,
                    banco TEXT,
                    monto TEXT,
                    fecha TEXT,
                    numero_comprobante TEXT,
                    cuenta_origen TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
            """)
            # Migration: Add metadata column if it doesn't exist
            try:
                conn.execute("ALTER TABLE pending_confirmations ADD COLUMN metadata TEXT")
            except sqlite3.OperationalError:
                pass # Already exists
    finally:
        conn.close()

def set_user_status(sender_jid, status):
    conn = get_db_connection()
    try:
        with conn:
            conn.execute("INSERT OR REPLACE INTO user_status (sender_jid, status, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)", (sender_jid, status))
    finally:
        conn.close()

def reset_user_status(sender_jid):
    """Resets user status back to 'bot'."""
    set_user_status(sender_jid, 'bot')

def get_user_status(sender_jid):
    conn = get_db_connection()
    try:
        row = conn.execute("SELECT status FROM user_status WHERE sender_jid = ?", (sender_jid,)).fetchone()
        return row['status'] if row else 'bot'
    finally:
        conn.close()

def save_pending_confirmation(sender_jid, data):
    conn = get_db_connection()
    try:
        metadata_json = json.dumps(data.get('metadata', {}))
        with conn:
            conn.execute("""
                INSERT OR REPLACE INTO pending_confirmations (sender_jid, message_id, output_path, original_filename, state, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (sender_jid, data['message_id'], data['output_path'], data['original_filename'], data.get('state', 'awaiting_confirmation'), metadata_json))
    finally:
        conn.close()

def get_pending_confirmation(sender_jid):
    conn = get_db_connection()
    try:
        row = conn.execute("SELECT * FROM pending_confirmations WHERE sender_jid = ?", (sender_jid,)).fetchone()
        if row:
            d = dict(row)
            try:
                d['metadata'] = json.loads(d['metadata']) if d.get('metadata') else {}
            except Exception:
                d['metadata'] = {}
            return d
        return None
    finally:
        conn.close()

def clear_pending_confirmation(sender_jid):
    conn = get_db_connection()
    try:
        with conn:
            cursor = conn.execute("DELETE FROM pending_confirmations WHERE sender_jid = ?", (sender_jid,))
            return cursor.rowcount > 0
    finally:
        conn.close()

def save_validated_registry(registry_data):
    conn = get_db_connection()
    try:
        with conn:
            conn.execute("""
                INSERT INTO validated_registries (unique_id, sender_jid, cedula, banco, monto, fecha, numero_comprobante, cuenta_origen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                registry_data['unique_id'], 
                registry_data['sender_jid'], 
                registry_data['cedula'], 
                registry_data['banco'], 
                registry_data['monto'], 
                registry_data['fecha'], 
                registry_data['numero_comprobante'], 
                registry_data['cuenta_origen']
            ))
    finally:
        conn.close()

def save_message(user_id, role, content):
    conn = get_db_connection()
    try:
        with conn:
            conn.execute("INSERT INTO conversations (user_id, role, content) VALUES (?, ?, ?)", (user_id, role, content))
    finally:
        conn.close()

def get_last_messages(user_id, limit=20):
    conn = get_db_connection()
    try:
        rows = conn.execute(
            "SELECT role, content FROM conversations WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit)
        ).fetchall()
        return [dict(r) for r in rows][::-1]
    finally:
        conn.close()
