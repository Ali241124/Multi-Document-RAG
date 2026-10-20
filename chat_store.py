
import json
import sqlite3
import uuid

from pathlib import Path


DB_PATH = Path(__file__).resolve().parent / "chat_history.sqlite3"


def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_db():
    """Create the chat and message tables if they do not exist."""
    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                collection_name TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chat_id)
                    REFERENCES chats(id)
                    ON DELETE CASCADE
            );
            """
        )


def create_chat(title="New Chat"):
    """Create a conversation with its own document collection."""
    chat_id = uuid.uuid4().hex
    collection_name = f"chat_{chat_id}"

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO chats (id, title, collection_name)
            VALUES (?, ?, ?)
            """,
            (chat_id, title, collection_name),
        )

    return get_chat(chat_id)


def list_chats():
    """Return saved conversations, newest activity first."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, title, collection_name, created_at, updated_at
            FROM chats
            ORDER BY updated_at DESC, created_at DESC
            """
        ).fetchall()

    return [dict(row) for row in rows]


def get_chat(chat_id):
    """Return one conversation, or None if it does not exist."""
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM chats WHERE id = ?",
            (chat_id,),
        ).fetchone()

    return dict(row) if row else None


def get_messages(chat_id):
    """Load a conversation's messages in chronological order."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT role, content, sources_json, created_at
            FROM messages
            WHERE chat_id = ?
            ORDER BY id ASC
            """,
            (chat_id,),
        ).fetchall()

    messages = []

    for row in rows:
        message = dict(row)
        message["sources"] = json.loads(
            message.pop("sources_json")
        )
        messages.append(message)

    return messages


def save_message(chat_id, role, content, sources=None):
    """Persist a user or assistant message."""
    if role not in ("user", "assistant"):
        raise ValueError("Role must be user or assistant.")

    sources_json = json.dumps(sources or [], ensure_ascii=False)

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO messages
                (chat_id, role, content, sources_json)
            VALUES (?, ?, ?, ?)
            """,
            (chat_id, role, content, sources_json),
        )

        connection.execute(
            """
            UPDATE chats
            SET updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (chat_id,),
        )


def rename_chat(chat_id, title):
    """Rename a saved conversation."""
    clean_title = title.strip()[:100] or "New Chat"

    with get_connection() as connection:
        connection.execute(
            """
            UPDATE chats
            SET title = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (clean_title, chat_id),
        )


def delete_chat(chat_id):
    """Delete a conversation and its saved messages."""
    with get_connection() as connection:
        connection.execute(
            "DELETE FROM chats WHERE id = ?",
            (chat_id,),
        )
