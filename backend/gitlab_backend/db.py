"""SQLite persistence for GitLab projects and issue runs."""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "gitlab.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _conn():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with _conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS gitlab_projects (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                project_url           TEXT    NOT NULL UNIQUE,
                gitlab_project_id     INTEGER NOT NULL,
                access_token          TEXT    NOT NULL,
                branch                TEXT    NOT NULL DEFAULT 'main',
                polling_interval_sec  INTEGER NOT NULL DEFAULT 60,
                csv_filename          TEXT    NOT NULL DEFAULT 'data.csv',
                instructions_filename TEXT    NOT NULL DEFAULT 'instructions.md',
                owner_label           TEXT,
                created_at            TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS gitlab_issue_runs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT    NOT NULL,
                gitlab_project_id INTEGER NOT NULL,
                issue_iid       INTEGER NOT NULL,
                issue_title     TEXT,
                triggered_by    TEXT,
                csv_filename    TEXT,
                task_type       TEXT,
                target_column   TEXT,
                started_at      TEXT    NOT NULL,
                completed_at    TEXT,
                status          TEXT    NOT NULL DEFAULT 'running',
                UNIQUE(gitlab_project_id, issue_iid)
            );
        """)


def upsert_project(
    project_url: str,
    gitlab_project_id: int,
    access_token: str,
    branch: str = "main",
    polling_interval_sec: int = 60,
    csv_filename: str = "data.csv",
    instructions_filename: str = "instructions.md",
    owner_label: str = "",
):
    with _conn() as conn:
        conn.execute("""
            INSERT INTO gitlab_projects
                (project_url, gitlab_project_id, access_token, branch,
                 polling_interval_sec, csv_filename, instructions_filename,
                 owner_label, created_at)
            VALUES (?,?,?,?,?,?,?,?,?)
            ON CONFLICT(project_url) DO UPDATE SET
                gitlab_project_id    = excluded.gitlab_project_id,
                access_token         = excluded.access_token,
                branch               = excluded.branch,
                polling_interval_sec = excluded.polling_interval_sec,
                csv_filename         = excluded.csv_filename,
                instructions_filename= excluded.instructions_filename,
                owner_label          = excluded.owner_label
        """, (
            project_url, gitlab_project_id, access_token, branch,
            polling_interval_sec, csv_filename, instructions_filename,
            owner_label or "", _now(),
        ))


def get_all_projects() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM gitlab_projects").fetchall()
    return [dict(r) for r in rows]


def try_claim_issue(
    session_id: str,
    gitlab_project_id: int,
    issue_iid: int,
    issue_title: str,
    triggered_by: str,
    csv_filename: str,
    task_type: str,
    target_column: str,
) -> bool:
    """Insert a run record atomically. Returns True if claimed, False if already exists."""
    try:
        with _conn() as conn:
            conn.execute("""
                INSERT INTO gitlab_issue_runs
                    (session_id, gitlab_project_id, issue_iid, issue_title,
                     triggered_by, csv_filename, task_type, target_column, started_at)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (
                session_id, gitlab_project_id, issue_iid, issue_title,
                triggered_by, csv_filename, task_type, target_column, _now(),
            ))
        return True
    except sqlite3.IntegrityError:
        return False


def update_run_status(session_id: str, status: str):
    completed = _now() if status in ("completed", "failed") else None
    with _conn() as conn:
        conn.execute("""
            UPDATE gitlab_issue_runs
            SET status = ?, completed_at = ?
            WHERE session_id = ?
        """, (status, completed, session_id))


def get_all_runs() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM gitlab_issue_runs ORDER BY started_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_run(session_id: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM gitlab_issue_runs WHERE session_id = ?", (session_id,)
        ).fetchone()
    return dict(row) if row else None
