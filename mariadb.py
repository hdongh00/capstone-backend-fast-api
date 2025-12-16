import os
import pymysql
from datetime import datetime, date
from typing import Optional, Dict, Any

class MariaAnalysisRepo:
    def __init__(self):
        self.host = os.getenv("MARIADB_HOST")
        self.port = int(os.getenv("MARIADB_PORT", "3306"))
        self.db = os.getenv("MARIADB_DB")
        self.user = os.getenv("MARIADB_USER")
        self.password = os.getenv("MARIADB_PASSWORD")

    def _conn(self):
        return pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.db,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True,
        )

    def get_latest_by_user_and_date(self, user_code: int, target_date: date) -> Optional[Dict[str, Any]]:
        sql = """
        SELECT analysis_code, user_code, emotion_score, emotion_name, summary, created_at
        FROM analysis_result
        WHERE user_code = %s AND DATE(created_at) = %s
        ORDER BY created_at DESC
        LIMIT 1
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (user_code, target_date.strftime("%Y-%m-%d")))
                row = cur.fetchone()
                return row

    def insert(self, user_code: int, emotion_score: float, emotion_name: str, summary: str, created_at: datetime) -> int:
        sql = """
        INSERT INTO analysis_result (user_code, emotion_score, emotion_name, summary, created_at)
        VALUES (%s, %s, %s, %s, %s)
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        user_code,
                        float(emotion_score),
                        (emotion_name or "")[:25],
                        (summary or "")[:3000],
                        created_at,
                    ),
                )
                return cur.lastrowid

    def update(self, analysis_code: int, emotion_score: float, emotion_name: str, summary: str, created_at: datetime) -> None:
        sql = """
        UPDATE analysis_result
        SET emotion_score=%s, emotion_name=%s, summary=%s, created_at=%s
        WHERE analysis_code=%s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        float(emotion_score),
                        (emotion_name or "")[:25],
                        (summary or "")[:3000],
                        created_at,
                        analysis_code,
                    ),
                )
