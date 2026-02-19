import sqlite3
import os

DB_PATH = "chat_history.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, timeout=20)
    conn.row_factory = sqlite3.Row
    # Enable WAL mode for better concurrent access in Gunicorn
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def init_db():
    with get_db_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS pending_confirmations (
                sender_jid TEXT PRIMARY KEY,
                message_id TEXT,
                output_path TEXT,
                original_filename TEXT,
                state TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()

def save_pending_confirmation(sender_jid, data):
    with get_db_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO pending_confirmations (sender_jid, message_id, output_path, original_filename, state)
            VALUES (?, ?, ?, ?, ?)
        """, (sender_jid, data['message_id'], data['output_path'], data['original_filename'], data.get('state', 'awaiting_confirmation')))
        conn.commit()

def get_pending_confirmation(sender_jid):
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM pending_confirmations WHERE sender_jid = ?", (sender_jid,)).fetchone()
        return dict(row) if row else None

def clear_pending_confirmation(sender_jid):
    with get_db_connection() as conn:
        cursor = conn.execute("DELETE FROM pending_confirmations WHERE sender_jid = ?", (sender_jid,))
        conn.commit()
        return cursor.rowcount > 0

def save_message(user_id, role, content):
    with get_db_connection() as conn:
        conn.execute("INSERT INTO conversations (user_id, role, content) VALUES (?, ?, ?)", (user_id, role, content))
        conn.commit()

def get_last_messages(user_id, limit=20):
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT role, content FROM conversations WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit)
        ).fetchall()
        return [dict(r) for r in rows][::-1]
