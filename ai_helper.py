from __future__ import annotations

import os
from typing import Optional
import pandas as pd


def heuristic_summary(df: pd.DataFrame) -> str:
    if df.empty:
        return "No incidents match the selected filters."

    total = len(df)
    open_count = int((df["status"] == "Open").sum())
    closed = df[df["status"] == "Closed"].copy()
    breach_rate = df["breached_sla"].mean()
    top_category = df["category"].value_counts().idxmax()
    top_team = df["assigned_team"].value_counts().idxmax()

    lines = [
        f"Analyzed {total} incidents. Open backlog currently stands at {open_count} incidents.",
        f"Overall SLA breach rate for the filtered slice is {breach_rate:.1%}.",
        f"The highest-volume category is {top_category}, and the busiest team is {top_team}.",
    ]

    if not closed.empty:
        breach_by_category = closed.groupby("category")["breached_sla"].mean().sort_values(ascending=False)
        worst_category = breach_by_category.index[0]
        lines.append(
            f"Among closed incidents, {worst_category} shows the highest breach rate at {breach_by_category.iloc[0]:.1%}."
        )

    lines.append(
        "Recommended action: focus on the highest-volume category, review backlog spikes, and prioritize patterns tied to long assignment delays and large alert volumes."
    )
    return " ".join(lines)


def llm_summary(df: pd.DataFrame) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("LLM_MODEL", "gpt-4.1-mini")
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    client = OpenAI(api_key=api_key)

    compact = df.head(30)[
        [
            "incident_id",
            "status",
            "severity",
            "category",
            "service",
            "assigned_team",
            "region",
            "alerts_count",
            "backlog_at_creation",
            "assignment_delay_hours",
            "breached_sla",
            "summary",
        ]
    ].to_dict(orient="records")

    prompt = (
        "You are summarizing enterprise incident operations data for a CIO analytics dashboard. "
        "Write a concise, business-friendly summary with the top operational risks, likely drivers, "
        "and 3 actionable recommendations. Here is a sample of the filtered data: "
        f"{compact}"
    )

    response = client.responses.create(
        model=model_name,
        input=prompt,
    )
    return response.output_text.strip()


def generate_ai_summary(df: pd.DataFrame) -> str:
    summary = llm_summary(df)
    if summary:
        return summary
    return heuristic_summary(df)
