from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.ai_helper import generate_ai_summary
from src.utils import apply_filters, load_data, load_metrics, load_model


st.set_page_config(page_title="Enterprise Incident Analytics Dashboard", layout="wide")

DATA_PATH = "data/incidents.csv"
MODEL_PATH = "models/sla_breach_model.joblib"
METRICS_PATH = "models/metrics.json"

df = load_data(DATA_PATH)
model = load_model(MODEL_PATH)
metrics = load_metrics(METRICS_PATH)

st.title("AI-Powered IT Operations Insights Dashboard")
st.caption("Synthetic enterprise incident dataset • SLA breach prediction • AI-assisted operational summaries")

min_date = df["created_at"].min().date()
max_date = df["created_at"].max().date()

with st.sidebar:
    st.header("Filters")
    selected_severity = st.multiselect("Severity", options=sorted(df["severity"].unique()))
    selected_category = st.multiselect("Category", options=sorted(df["category"].unique()))
    selected_team = st.multiselect("Assigned Team", options=sorted(df["assigned_team"].unique()))
    selected_region = st.multiselect("Region", options=sorted(df["region"].unique()))
    selected_bu = st.multiselect("Business Unit", options=sorted(df["business_unit"].unique()))
    selected_dates = st.date_input("Created date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    date_range = selected_dates
else:
    date_range = (min_date, max_date)

filtered = apply_filters(
    df,
    severity=selected_severity,
    category=selected_category,
    assigned_team=selected_team,
    region=selected_region,
    business_unit=selected_bu,
    date_range=date_range,
)

overview_tab, explorer_tab, prediction_tab, insights_tab = st.tabs(
    ["Overview", "Explorer", "Prediction", "AI Insights"]
)

with overview_tab:
    total_incidents = len(filtered)
    open_incidents = int((filtered["status"] == "Open").sum())
    closed_filtered = filtered[filtered["status"] == "Closed"].copy()
    breach_rate = float(filtered["breached_sla"].mean()) if total_incidents else 0.0
    avg_resolution = (
        float(closed_filtered["resolution_hours"].mean()) if not closed_filtered.empty else 0.0
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Incidents", f"{total_incidents:,}")
    c2.metric("Open Backlog", f"{open_incidents:,}")
    c3.metric("Breach Rate", f"{breach_rate:.1%}")
    c4.metric("Avg Resolution (Closed)", f"{avg_resolution:.2f} hrs")

    model_cols = st.columns(5)
    model_cols[0].metric("Model Accuracy", f"{metrics['accuracy']:.2%}")
    model_cols[1].metric("Precision", f"{metrics['precision']:.2%}")
    model_cols[2].metric("Recall", f"{metrics['recall']:.2%}")
    model_cols[3].metric("F1", f"{metrics['f1']:.2%}")
    model_cols[4].metric("ROC-AUC", f"{metrics['roc_auc']:.2%}")

    if total_incidents:
        monthly = filtered.groupby("month").size().reset_index(name="incidents")
        fig_month = px.line(monthly, x="month", y="incidents", markers=True, title="Incident Volume by Month")
        st.plotly_chart(fig_month, use_container_width=True)

        by_category = filtered.groupby("category").size().reset_index(name="incidents").sort_values("incidents", ascending=False)
        fig_cat = px.bar(by_category, x="category", y="incidents", title="Incident Volume by Category")
        st.plotly_chart(fig_cat, use_container_width=True)

with explorer_tab:
    st.subheader("Breach and resolution diagnostics")

    if not filtered.empty:
        by_team = filtered.groupby("assigned_team").size().reset_index(name="incidents").sort_values("incidents", ascending=False)
        fig_team = px.bar(by_team, x="assigned_team", y="incidents", title="Incident Volume by Team")
        st.plotly_chart(fig_team, use_container_width=True)

        closed_view = filtered[filtered["status"] == "Closed"].copy()
        if not closed_view.empty:
            breach_by_category = (
                closed_view.groupby("category")["breached_sla"].mean().reset_index(name="breach_rate")
            )
            fig_breach = px.bar(
                breach_by_category,
                x="category",
                y="breach_rate",
                title="Closed-Incident SLA Breach Rate by Category",
            )
            st.plotly_chart(fig_breach, use_container_width=True)

            fig_scatter = px.scatter(
                closed_view,
                x="assignment_delay_hours",
                y="resolution_hours",
                color="severity",
                hover_data=["incident_id", "category", "assigned_team"],
                title="Assignment Delay vs Resolution Time",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Incident records")
    st.dataframe(
        filtered[
            [
                "incident_id",
                "created_at",
                "status",
                "severity",
                "category",
                "service",
                "assigned_team",
                "region",
                "alerts_count",
                "backlog_at_creation",
                "breached_sla",
                "summary",
            ]
        ].sort_values("created_at", ascending=False),
        use_container_width=True,
        height=420,
    )

with prediction_tab:
    st.subheader("Predict SLA breach risk for a new incident")

    col1, col2, col3 = st.columns(3)
    with col1:
        p_severity = st.selectbox("Severity", sorted(df["severity"].unique()))
        p_category = st.selectbox("Category", sorted(df["category"].unique()))
        p_service = st.selectbox("Service", sorted(df["service"].unique()))
        p_team = st.selectbox("Assigned Team", sorted(df["assigned_team"].unique()))
        p_region = st.selectbox("Region", sorted(df["region"].unique()))
    with col2:
        p_bu = st.selectbox("Business Unit", sorted(df["business_unit"].unique()))
        p_alerts = st.slider("Alerts Count", 0, 20, 3)
        p_backlog = st.slider("Backlog at Creation", 0, 80, 25)
        p_repeat = st.selectbox("Repeat Incident", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with col3:
        p_users = st.slider("Affected Users", 1, 200, 15)
        p_assignment_delay = st.slider("Assignment Delay (hours)", 0.0, 12.0, 1.0, step=0.25)
        p_comments = st.slider("Comments", 0, 25, 5)
        p_hour = st.slider("Created Hour", 0, 23, 10)
        p_weekend = st.selectbox("Weekend", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    input_df = pd.DataFrame(
        [
            {
                "severity": p_severity,
                "category": p_category,
                "service": p_service,
                "assigned_team": p_team,
                "region": p_region,
                "business_unit": p_bu,
                "alerts_count": p_alerts,
                "backlog_at_creation": p_backlog,
                "repeat_incident": p_repeat,
                "affected_users": p_users,
                "assignment_delay_hours": p_assignment_delay,
                "num_comments": p_comments,
                "created_hour": p_hour,
                "is_weekend": p_weekend,
            }
        ]
    )

    probability = float(model.predict_proba(input_df)[0, 1])
    prediction = int(probability >= 0.50)

    risk_label = "High Risk" if prediction == 1 else "Lower Risk"
    st.metric("Predicted SLA Breach Probability", f"{probability:.1%}", risk_label)

    if probability >= 0.65:
        st.warning("This incident is likely to breach SLA. Consider faster assignment, queue balancing, or escalation.")
    elif probability >= 0.40:
        st.info("This incident shows moderate risk. Monitor backlog and response time closely.")
    else:
        st.success("This incident currently looks less likely to breach SLA.")

with insights_tab:
    st.subheader("AI-generated operational summary")
    st.write(
        "This tab uses a local heuristic summary by default. If you provide `OPENAI_API_KEY` in your environment, "
        "the dashboard will attempt to generate an LLM-based summary."
    )

    if st.button("Generate Summary"):
        summary_text = generate_ai_summary(filtered)
        st.text_area("Summary", value=summary_text, height=220)

        if not filtered.empty:
            top_drivers = (
                filtered.groupby("category")
                .agg(incidents=("incident_id", "count"), breach_rate=("breached_sla", "mean"))
                .sort_values(["incidents", "breach_rate"], ascending=False)
                .head(5)
            )
            st.dataframe(top_drivers, use_container_width=True)
