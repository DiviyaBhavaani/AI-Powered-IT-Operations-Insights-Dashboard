from __future__ import annotations

from pathlib import Path
import sqlite3
import pandas as pd


def create_sqlite_database(csv_path: str | Path = "data/incidents.csv", db_path: str | Path = "data/incidents.db") -> Path:
    csv_path = Path(csv_path)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    with sqlite3.connect(db_path) as connection:
        df.to_sql("incidents", connection, if_exists="replace", index=False)

    return db_path


if __name__ == "__main__":
    output = create_sqlite_database()
    print(f"Created SQLite database at {output}")
