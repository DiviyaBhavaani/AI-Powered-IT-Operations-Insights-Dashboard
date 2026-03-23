from __future__ import annotations

from src.create_database import create_sqlite_database
from src.eda_report import build_eda_report
from src.generate_data import save_dataset
from src.train_model import train_model


def main() -> None:
    save_dataset("data/incidents.csv")
    create_sqlite_database("data/incidents.csv", "data/incidents.db")
    metrics = train_model("data/incidents.csv", "models")
    build_eda_report("data/incidents.csv", "reports")
    print("Pipeline complete.")
    print(metrics)


if __name__ == "__main__":
    main()
