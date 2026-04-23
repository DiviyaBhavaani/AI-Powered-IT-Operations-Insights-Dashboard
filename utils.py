from __future__ import annotations

from pathlib import Path
import json
import joblib
import pandas as pd


def load_data(data_path: str | Path = "data/incidents.csv") -> pd.DataFrame:
    df = pd.read_csv(data_path, parse_dates=["created_at", "resolved_at"])
    return df


def load_model(model_path: str | Path = "models/sla_breach_model.joblib"):
    return joblib.load(model_path)


def load_metrics(metrics_path: str | Path = "models/metrics.json") -> dict:
    with open(metrics_path, "r", encoding="utf-8") as file:
        return json.load(file)


def apply_filters(
    df: pd.DataFrame,
    severity: list[str],
    category: list[str],
    assigned_team: list[str],
    region: list[str],
    business_unit: list[str],
    date_range: tuple[pd.Timestamp, pd.Timestamp],
) -> pd.DataFrame:
    filtered = df.copy()
    if severity:
        filtered = filtered[filtered["severity"].isin(severity)]
    if category:
        filtered = filtered[filtered["category"].isin(category)]
    if assigned_team:
        filtered = filtered[filtered["assigned_team"].isin(assigned_team)]
    if region:
        filtered = filtered[filtered["region"].isin(region)]
    if business_unit:
        filtered = filtered[filtered["business_unit"].isin(business_unit)]

    start_date, end_date = date_range
    filtered = filtered[(filtered["created_at"] >= pd.Timestamp(start_date)) & (filtered["created_at"] <= pd.Timestamp(end_date))]
    return filtered
