from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class GenerationConfig:
    n_rows: int = 3500
    seed: int = 42
    start_date: str = "2025-01-01"
    end_date: str = "2026-03-15"


def generate_incident_data(config: GenerationConfig = GenerationConfig()) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    start = pd.Timestamp(config.start_date)
    end = pd.Timestamp(config.end_date)

    categories = ["Network", "Cloud", "Database", "Security", "Application", "Access"]
    services = {
        "Network": ["SD-WAN", "LAN", "VPN", "Firewall"],
        "Cloud": ["Azure", "AWS", "VMware", "Kubernetes"],
        "Database": ["PostgreSQL", "Oracle", "MySQL", "MongoDB"],
        "Security": ["IAM", "SIEM", "Endpoint", "Vulnerability Mgmt"],
        "Application": ["ERP", "CRM", "Billing", "Middleware"],
        "Access": ["SSO", "VPN Access", "Privileged Access", "Email Access"],
    }
    teams = {
        "Network": ["NetOps", "NOC", "Infra Operations"],
        "Cloud": ["CloudOps", "Platform Engineering", "SRE"],
        "Database": ["DBA", "Platform Engineering", "SRE"],
        "Security": ["SecOps", "IAM Operations", "SOC"],
        "Application": ["App Support", "Middleware Ops", "ERP Support"],
        "Access": ["IAM Operations", "Service Desk", "EUC Support"],
    }
    regions = ["North America", "Europe", "APAC", "LATAM"]
    business_units = ["Finance", "HR", "Sales", "Operations", "Engineering", "Support"]
    severity_levels = ["P1", "P2", "P3", "P4"]
    severity_probs = [0.08, 0.18, 0.42, 0.32]
    sla_map = {"P1": 4, "P2": 8, "P3": 24, "P4": 72}
    seconds_range = int((end - start).total_seconds())

    rows: list[dict] = []

    for i in range(1, config.n_rows + 1):
        created = start + pd.Timedelta(seconds=int(rng.integers(0, seconds_range)))
        month_num = created.month

        if month_num in [11, 12, 1, 2]:
            category = rng.choice(categories, p=[0.20, 0.24, 0.13, 0.16, 0.17, 0.10])
        elif month_num in [6, 7, 8]:
            category = rng.choice(categories, p=[0.16, 0.18, 0.15, 0.14, 0.25, 0.12])
        else:
            category = rng.choice(categories, p=[0.18, 0.20, 0.14, 0.15, 0.22, 0.11])

        service = rng.choice(services[category])
        assigned_team = rng.choice(teams[category])
        severity = rng.choice(severity_levels, p=severity_probs)
        region = rng.choice(regions, p=[0.38, 0.24, 0.28, 0.10])
        business_unit = rng.choice(business_units)

        alerts_count = int(max(0, rng.poisson({
            "P1": 6, "P2": 4, "P3": 2, "P4": 1
        }[severity] + (1 if category in ["Cloud", "Network"] and month_num in [11, 12, 1, 2] else 0))))

        backlog_at_creation = int(max(0, rng.normal(24 + (4 if month_num in [1, 2, 11, 12] else 0), 10)))
        repeat_incident = int(
            rng.random() < {
                "P1": 0.26, "P2": 0.21, "P3": 0.14, "P4": 0.08
            }[severity] + (0.06 if category in ["Application", "Access"] else 0)
        )
        affected_users = int(max(1, rng.lognormal(
            mean={"P1": 4.3, "P2": 3.6, "P3": 2.9, "P4": 2.1}[severity],
            sigma=0.52,
        )))
        assignment_delay_hours = max(0, rng.normal(
            {"P1": 0.25, "P2": 0.55, "P3": 1.8, "P4": 6}[severity],
            {"P1": 0.15, "P2": 0.35, "P3": 0.9, "P4": 2.5}[severity],
        ))
        num_comments = int(max(0, rng.poisson({"P1": 13, "P2": 8, "P3": 5, "P4": 3}[severity])))
        created_hour = int(created.hour)
        is_weekend = int(created.weekday() >= 5)

        base_hours = {"P1": 2.6, "P2": 5.2, "P3": 14.5, "P4": 39}[severity]
        category_adjustment = {
            "Network": 1.0,
            "Cloud": 1.16,
            "Database": 1.10,
            "Security": 1.04,
            "Application": 1.24,
            "Access": 1.08,
        }[category]
        workload_factor = 1 + backlog_at_creation / 220 + alerts_count / 45 + repeat_incident * 0.18 + is_weekend * 0.10
        off_hours_factor = 1.14 if (created_hour < 7 or created_hour > 20) else 1.0
        team_noise = rng.normal(1.0, 0.18)

        resolution_hours = max(
            0.5,
            base_hours * category_adjustment * workload_factor * off_hours_factor * team_noise + assignment_delay_hours * 0.75,
        )

        is_open = rng.random() < 0.16
        if is_open:
            current_age_hours = max(0.25, (end - created).total_seconds() / 3600.0)
            resolved_at = pd.NaT
            status = "Open"
            breached_sla = int(current_age_hours > sla_map[severity])
            resolution_hours_value = np.nan
            current_age_value = current_age_hours
        else:
            resolved_at = created + pd.Timedelta(hours=float(resolution_hours))
            status = "Closed"
            breached_sla = int(resolution_hours > sla_map[severity])
            resolution_hours_value = float(resolution_hours)
            current_age_value = float(resolution_hours)

        root_cause = {
            "Network": rng.choice(["Link instability", "Routing drift", "Firewall rule error", "Capacity saturation"]),
            "Cloud": rng.choice(["Autoscaling lag", "Storage latency", "Node pressure", "Misconfigured deployment"]),
            "Database": rng.choice(["Slow queries", "Replication lag", "Lock contention", "Connection pool exhaustion"]),
            "Security": rng.choice(["Policy misconfiguration", "Expired certificate", "Suspicious login pattern", "Endpoint isolation event"]),
            "Application": rng.choice(["Release defect", "API timeout", "Dependency failure", "Cache inconsistency"]),
            "Access": rng.choice(["Provisioning delay", "SSO token issue", "Permission mismatch", "Mailbox sync error"]),
        }[category]

        summary = (
            f"{category} issue affecting {service} in {region}; priority {severity}; "
            f"root cause: {root_cause}; team {assigned_team}"
        )

        rows.append(
            {
                "incident_id": f"INC{i:06d}",
                "created_at": created,
                "resolved_at": resolved_at,
                "status": status,
                "severity": severity,
                "category": category,
                "service": service,
                "assigned_team": assigned_team,
                "region": region,
                "business_unit": business_unit,
                "root_cause": root_cause,
                "alerts_count": alerts_count,
                "backlog_at_creation": backlog_at_creation,
                "repeat_incident": repeat_incident,
                "affected_users": affected_users,
                "assignment_delay_hours": round(assignment_delay_hours, 2),
                "num_comments": num_comments,
                "sla_hours": sla_map[severity],
                "current_age_hours": round(current_age_value, 2),
                "resolution_hours": round(resolution_hours_value, 2) if not pd.isna(resolution_hours_value) else np.nan,
                "breached_sla": breached_sla,
                "created_hour": created_hour,
                "is_weekend": is_weekend,
                "month": created.strftime("%Y-%m"),
                "summary": summary,
            }
        )

    return pd.DataFrame(rows)


def save_dataset(output_path: str | Path = "data/incidents.csv", config: GenerationConfig = GenerationConfig()) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_incident_data(config)
    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    path = save_dataset()
    print(f"Saved dataset to {path}")
