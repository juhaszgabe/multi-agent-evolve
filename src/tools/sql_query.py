import sqlite3
import time
import pandas as pd
from .base import ToolResult


def sql_query(query: str, csv_path: str) -> ToolResult:
    start = time.time()
    try:
        df = pd.read_csv(csv_path) if csv_path.lower().endswith(".csv") else pd.read_excel(csv_path)
        conn = sqlite3.connect(":memory:")
        df.to_sql("data", conn, index=False, if_exists="replace")
        cursor = conn.execute(query)
        columns = [desc[0] for desc in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        return ToolResult(
            success=True,
            output={"rows": rows, "columns": columns},
            latency_ms=(time.time() - start) * 1000,
        )
    except Exception as e:
        return ToolResult(success=False, output=None, error=str(e), latency_ms=(time.time() - start) * 1000)
