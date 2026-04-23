from __future__ import annotations

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_eda_report(data_path: str | Path = "data/incidents.csv", report_dir: str | Path = "reports") -> dict:
    data_path = Path(data_path)
    report_dir = Path(report_dir)
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path, parse_dates=["created_at", "resolved_at"])

    monthly = df.groupby("month").size().reset_index(name="incidents")
    closed = df[df["status"] == "Closed"].copy()

    breach_by_category = closed.groupby("category")["breached_sla"].mean().sort_values(ascending=False)
    resolution_by_severity = closed.groupby("severity")["resolution_hours"].mean().reindex(["P1", "P2", "P3", "P4"])
    open_by_team = df[df["status"] == "Open"].groupby("assigned_team").size().sort_values(ascending=False).head(10)
    breach_by_region = closed.groupby("region")["breached_sla"].mean().sort_values(ascending=False)

    plt.figure(figsize=(10, 4))
    plt.plot(monthly["month"], monthly["incidents"], marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.title("Monthly Incident Volume")
    plt.tight_layout()
    plt.savefig(figure_dir / "monthly_incident_volume.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    breach_by_category.plot(kind="bar")
    plt.ylabel("Breach Rate")
    plt.title("Closed-Incident SLA Breach Rate by Category")
    plt.tight_layout()
    plt.savefig(figure_dir / "breach_rate_by_category.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    resolution_by_severity.plot(kind="bar")
    plt.ylabel("Average Resolution Hours")
    plt.title("Average Resolution Time by Severity")
    plt.tight_layout()
    plt.savefig(figure_dir / "resolution_by_severity.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    open_by_team.sort_values().plot(kind="barh")
    plt.xlabel("Open Incidents")
    plt.title("Top Teams by Open Incident Count")
    plt.tight_layout()
    plt.savefig(figure_dir / "open_incidents_by_team.png", dpi=160)
    plt.close()

    summary = {
        "total_incidents": int(len(df)),
        "open_incidents": int((df["status"] == "Open").sum()),
        "closed_incidents": int((df["status"] == "Closed").sum()),
        "overall_breach_rate": float(df["breached_sla"].mean()),
        "closed_breach_rate": float(closed["breached_sla"].mean()),
        "avg_resolution_closed_hours": float(closed["resolution_hours"].mean()),
        "median_resolution_closed_hours": float(closed["resolution_hours"].median()),
        "highest_breach_category": str(breach_by_category.idxmax()),
        "highest_breach_category_rate": float(breach_by_category.max()),
        "lowest_breach_category": str(breach_by_category.idxmin()),
        "lowest_breach_category_rate": float(breach_by_category.min()),
        "top_open_team": str(open_by_team.index[0]),
        "top_open_team_count": int(open_by_team.iloc[0]),
        "top_region_breach": str(breach_by_region.idxmax()),
        "top_region_breach_rate": float(breach_by_region.max()),
    }

    with open(report_dir / "analysis_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    markdown = f"""# Exploratory Data Analysis

## Dataset at a glance
- Total incidents: **{summary['total_incidents']:,}**
- Closed incidents: **{summary['closed_incidents']:,}**
- Open incidents: **{summary['open_incidents']:,}**
- Overall SLA breach rate: **{summary['overall_breach_rate']:.1%}**
- Closed-incident SLA breach rate: **{summary['closed_breach_rate']:.1%}**
- Average closed-incident resolution time: **{summary['avg_resolution_closed_hours']:.2f} hours**
- Median closed-incident resolution time: **{summary['median_resolution_closed_hours']:.2f} hours**

## Key findings
1. **{summary['highest_breach_category']}** has the highest closed-incident breach rate at **{summary['highest_breach_category_rate']:.1%}**.
2. **{summary['lowest_breach_category']}** has the lowest closed-incident breach rate at **{summary['lowest_breach_category_rate']:.1%}**.
3. The team with the largest open backlog is **{summary['top_open_team']}** with **{summary['top_open_team_count']}** open incidents.
4. **{summary['top_region_breach']}** has the highest closed-incident breach rate among regions at **{summary['top_region_breach_rate']:.1%}**.

## Generated figures
- `figures/monthly_incident_volume.png`
- `figures/breach_rate_by_category.png`
- `figures/resolution_by_severity.png`
- `figures/open_incidents_by_team.png`

## Notes
This dataset is synthetic and designed for portfolio use. The analysis is intended to demonstrate a realistic enterprise incident-management workflow without using any confidential production data.
"""
    (report_dir / "eda_summary.md").write_text(markdown, encoding="utf-8")
    return summary


if __name__ == "__main__":
    output = build_eda_report()
    print(json.dumps(output, indent=2))
