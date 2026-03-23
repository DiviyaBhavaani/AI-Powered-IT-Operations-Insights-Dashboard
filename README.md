# AI Powered IT Operations Insights Dashboard


## Project objectives

This project demonstrates how to:

- clean and analyze incident management data
- create executive style metrics and trend views
- predict whether an incident is likely to breach SLA
- surface operational bottlenecks by category, region, and team
- generate concise AI-assisted summaries for decision support

---

## Repository structure

```text
.
├── app.py
├── data
│   ├── incidents.csv
│   └── incidents.db
├── models
│   ├── metrics.json
│   ├── sla_breach_model.joblib
│   └── top_features.csv
├── notebooks
│   └── eda_walkthrough.ipynb
├── reports
│   ├── analysis_summary.json
│   ├── eda_summary.md
│   ├── model_report.md
│   └── figures
├── run_pipeline.py
├── sql
│   └── analysis_queries.sql
└── src
    ├── ai_helper.py
    ├── create_database.py
    ├── eda_report.py
    ├── generate_data.py
    ├── train_model.py
    └── utils.py
```

---

## Dataset

The dataset contains synthetic enterprise-style incidents with the following fields:

- incident ID, timestamps, status
- severity, category, service, assigned team
- region, business unit, root cause
- alert count, backlog size, affected users
- assignment delay, number of comments
- SLA target and breach flag

---

## Key analytics questions answered

1. Which categories drive the highest incident volume?
2. Which categories and teams have the worst SLA performance?
3. Which teams are carrying the highest open backlog?
4. Which features are most predictive of SLA breach?
5. How can an AI summary convert raw operational data into business-friendly recommendations?

---

## Model

The project trains a **Random Forest classifier** to predict whether a closed incident will breach SLA.

### Input features
- severity
- category
- service
- assigned team
- region
- business unit
- alerts count
- backlog at creation
- repeat incident flag
- affected users
- assignment delay hours
- number of comments
- created hour
- weekend flag

### Saved outputs
- trained model: `models/sla_breach_model.joblib`
- metrics: `models/metrics.json`
- feature importances: `models/top_features.csv`

---

## Dashboard features

The Streamlit app includes:

### 1. Overview
- incident count
- open backlog
- breach rate
- average resolution time
- model performance metrics
- monthly incident trend
- incident volume by category

### 2. Explorer
- team-level incident views
- category-level breach rate
- assignment delay vs. resolution time
- raw filtered incident records

### 3. Prediction
- interactive SLA breach risk scoring for a new incident

### 4. AI Insights
- local heuristic summary
- optional LLM-generated summary if `OPENAI_API_KEY` is available

---

## How to run locally

### 1) Create environment and install dependencies
```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### 2) Rebuild the pipeline
```bash
python run_pipeline.py
```

### 3) Launch the dashboard
```bash
streamlit run app.py
```

---

## Optional LLM setup

To enable the LLM summary in the AI Insights tab, set:

```bash
export OPENAI_API_KEY="your_api_key_here"
export LLM_MODEL="gpt-4.1-mini"
```

If no API key is present, the app falls back to a local rules-based summary so the project still runs end-to-end.

---

## Example project outcomes

This repo already includes:
- a generated dataset (`data/incidents.csv`)
- a SQLite database (`data/incidents.db`)
- a trained model
- EDA figures
- a markdown EDA report
- a model performance report

---

## Suggested resume entry

**AI-Powered IT Operations Insights Dashboard** | Python, SQL, Streamlit, Scikit-learn, Plotly, SQLite  
- Built a synthetic enterprise incident analytics platform to track SLA performance, backlog health, and operational trends across categories, regions, and support teams.  
- Trained a machine-learning model to predict SLA breach risk using incident metadata, workload signals, and response-timing features; surfaced model metrics and key drivers in a dashboard.  
- Added AI-assisted summaries to convert filtered incident data into business-friendly operational insights and actionable recommendations for decision makers.  

---

## Notes

- This project is for portfolio and interview demonstration.
- The data is synthetic and does **not** represent any real Kyndryl, client, or production environment.
