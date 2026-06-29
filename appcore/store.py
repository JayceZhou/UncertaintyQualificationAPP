"""SQLite persistence for accounts, projects and audit events."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import sqlite3

from .security import hash_password, validate_username, verify_password


@dataclass(frozen=True)
class User:
    id: int
    username: str
    display_name: str
    role: str
    status: str
    created_at: str


@dataclass(frozen=True)
class Project:
    id: int
    owner_id: int
    name: str
    code: str
    task_type: str
    description: str
    status: str
    created_at: str


class AppStore:
    """Small local database suitable for the single-node V1.0 application."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    display_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('admin', 'user')),
                    status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'disabled')),
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    code TEXT NOT NULL,
                    task_type TEXT NOT NULL CHECK(task_type IN ('classification', 'vector_field')),
                    description TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'draft' CHECK(status IN ('draft', 'completed')),
                    created_at TEXT NOT NULL,
                    UNIQUE(owner_id, code)
                );
                CREATE TABLE IF NOT EXISTS operation_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                    action TEXT NOT NULL,
                    target_type TEXT NOT NULL,
                    target_id TEXT,
                    details TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_projects_owner ON projects(owner_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_logs_created ON operation_logs(created_at DESC);
                """
            )

    @staticmethod
    def _user(row: sqlite3.Row | None) -> User | None:
        if row is None:
            return None
        return User(
            id=row["id"],
            username=row["username"],
            display_name=row["display_name"],
            role=row["role"],
            status=row["status"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _project(row: sqlite3.Row) -> Project:
        return Project(
            id=row["id"],
            owner_id=row["owner_id"],
            name=row["name"],
            code=row["code"],
            task_type=row["task_type"],
            description=row["description"],
            status=row["status"],
            created_at=row["created_at"],
        )

    def register(self, username: str, display_name: str, password: str) -> User:
        username = validate_username(username)
        display_name = display_name.strip() or username
        if len(display_name) > 40:
            raise ValueError("显示名称不能超过 40 个字符")
        password_hash = hash_password(password)
        with self._connect() as connection:
            # Serialize first-account role assignment so concurrent first-time
            # registrations cannot both become administrators.
            connection.execute("BEGIN IMMEDIATE")
            role = "admin" if connection.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0 else "user"
            try:
                cursor = connection.execute(
                    "INSERT INTO users(username, display_name, password_hash, role, created_at) VALUES(?,?,?,?,?)",
                    (username, display_name, password_hash, role, self._now()),
                )
            except sqlite3.IntegrityError as error:
                raise ValueError("用户名已存在") from error
            user_id = cursor.lastrowid
            self._log(connection, user_id, "register", "user", str(user_id), {"role": role})
            row = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return self._user(row)  # type: ignore[return-value]

    def authenticate(self, username: str, password: str) -> User | None:
        normalized = username.strip()
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM users WHERE username = ?", (normalized,)).fetchone()
        if row is None or row["status"] != "active" or not verify_password(password, row["password_hash"]):
            return None
        return self._user(row)

    def get_user(self, user_id: int) -> User | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return self._user(row)

    def list_users(self) -> list[User]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM users ORDER BY created_at, id").fetchall()
        return [self._user(row) for row in rows if row is not None]  # type: ignore[misc]

    def set_user_status(self, actor: User, user_id: int, status: str) -> None:
        if actor.role != "admin":
            raise PermissionError("仅管理员可修改用户状态")
        if user_id == actor.id:
            raise ValueError("不能停用当前登录账户")
        if status not in {"active", "disabled"}:
            raise ValueError("无效用户状态")
        with self._connect() as connection:
            cursor = connection.execute("UPDATE users SET status = ? WHERE id = ?", (status, user_id))
            if cursor.rowcount != 1:
                raise ValueError("用户不存在")
            self._log(connection, actor.id, "set_user_status", "user", str(user_id), {"status": status})

    def change_password(self, user: User, current_password: str, new_password: str) -> None:
        if self.authenticate(user.username, current_password) is None:
            raise ValueError("当前密码不正确")
        password_hash = hash_password(new_password)
        with self._connect() as connection:
            connection.execute("UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user.id))
            self._log(connection, user.id, "change_password", "user", str(user.id), {})

    def create_project(
        self,
        owner: User,
        name: str,
        code: str,
        task_type: str,
        description: str = "",
    ) -> Project:
        name = name.strip()
        code = code.strip().upper()
        description = description.strip()
        if not (2 <= len(name) <= 60):
            raise ValueError("项目名称须为 2-60 个字符")
        if not code or len(code) > 24 or not all(character.isalnum() or character in "-_" for character in code):
            raise ValueError("项目编号仅允许 1-24 位字母、数字、短横线或下划线")
        if task_type not in {"classification", "vector_field"}:
            raise ValueError("无效任务类型")
        with self._connect() as connection:
            try:
                cursor = connection.execute(
                    "INSERT INTO projects(owner_id,name,code,task_type,description,created_at) VALUES(?,?,?,?,?,?)",
                    (owner.id, name, code, task_type, description, self._now()),
                )
            except sqlite3.IntegrityError as error:
                raise ValueError("项目编号已存在") from error
            project_id = cursor.lastrowid
            self._log(connection, owner.id, "create", "project", str(project_id), {"code": code})
            row = connection.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return self._project(row)

    def list_projects(self, owner: User) -> list[Project]:
        query = "SELECT * FROM projects"
        parameters: tuple = ()
        if owner.role != "admin":
            query += " WHERE owner_id = ?"
            parameters = (owner.id,)
        query += " ORDER BY created_at DESC, id DESC"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._project(row) for row in rows]

    def delete_project(self, actor: User, project_id: int) -> None:
        with self._connect() as connection:
            row = connection.execute("SELECT owner_id, code FROM projects WHERE id = ?", (project_id,)).fetchone()
            if row is None:
                raise ValueError("项目不存在")
            if actor.role != "admin" and row["owner_id"] != actor.id:
                raise PermissionError("无权删除该项目")
            connection.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            self._log(connection, actor.id, "delete", "project", str(project_id), {"code": row["code"]})

    def log(self, user_id: int | None, action: str, target_type: str, target_id: str | None = None, details=None) -> None:
        with self._connect() as connection:
            self._log(connection, user_id, action, target_type, target_id, details or {})

    def _log(self, connection, user_id, action, target_type, target_id, details) -> None:
        connection.execute(
            "INSERT INTO operation_logs(user_id,action,target_type,target_id,details,created_at) VALUES(?,?,?,?,?,?)",
            (user_id, action, target_type, target_id, json.dumps(details, ensure_ascii=False), self._now()),
        )

    def list_logs(self, limit: int = 100) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT l.id, l.action, l.target_type, l.target_id, l.details, l.created_at,
                       COALESCE(u.username, 'system') AS username
                FROM operation_logs l LEFT JOIN users u ON u.id = l.user_id
                ORDER BY l.id DESC LIMIT ?
                """,
                (max(1, min(limit, 500)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def dashboard_stats(self, user: User) -> dict[str, int]:
        with self._connect() as connection:
            if user.role == "admin":
                project_count = connection.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
                user_count = connection.execute("SELECT COUNT(*) FROM users WHERE status='active'").fetchone()[0]
            else:
                project_count = connection.execute("SELECT COUNT(*) FROM projects WHERE owner_id=?", (user.id,)).fetchone()[0]
                user_count = 1
            log_count = connection.execute("SELECT COUNT(*) FROM operation_logs WHERE user_id=?", (user.id,)).fetchone()[0]
        return {"projects": project_count, "active_users": user_count, "operations": log_count}
