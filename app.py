from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from data_tools import DataValidationError, PremiumDataStore
from premium_agent import PremiumVarianceAgent


st.set_page_config(page_title="Premium Variance AI Agent V2", page_icon="📊", layout="wide")
st.title("📊 Premium Variance AI Agent — V2")
st.caption(
    "Gemini understands the question and writes commentary. Python performs every lookup, filter, calculation, comparison and ranking."
)

# Copy secrets into environment variables used by Agno/Gemini.
try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    if "GEMINI_MODEL" in st.secrets:
        os.environ["GEMINI_MODEL"] = st.secrets["GEMINI_MODEL"]
except Exception:
    pass

if not os.getenv("GOOGLE_API_KEY"):
    st.error(
        "GOOGLE_API_KEY is missing. Copy .streamlit/secrets.toml.example to .streamlit/secrets.toml and add your Gemini API key."
    )
    st.stop()

DEFAULT_FILE = Path(__file__).parent / "data" / "Premium Variance Data.xlsx"
uploaded_file = st.sidebar.file_uploader("Upload a Premium Variance workbook", type=["xlsx"])
source = uploaded_file if uploaded_file is not None else DEFAULT_FILE

try:
    data_store = PremiumDataStore(source)
except DataValidationError as exc:
    st.error(f"Workbook validation failed: {exc}")
    st.stop()

model_id = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
agent = PremiumVarianceAgent(data_store, model_id=model_id)

stats = data_store.dataset_stats
st.sidebar.success(
    f"Dataset ready: {stats['Rows']} rows | {stats['Clients']} clients | {stats['Treaties']} treaties | {stats['Quarters']} quarters"
)
st.sidebar.caption(f"Gemini model: {model_id}")
st.sidebar.info("Using uploaded workbook." if uploaded_file else "Using the included 5,000-row mock workbook.")

with st.sidebar.expander("Example questions", expanded=True):
    st.write("• Explain Client 001")
    st.write("• Explain Treaty 10001")
    st.write("• How did Term portfolio perform in Q3?")
    st.write("• Show age 45 across all quarters")
    st.write("• Compare Client 001 between Q1 and Q4")
    st.write("• Compare Group between Q2 and Q3")
    st.write("• Which 10 treaties had the worst persistency in Q4?")
    st.write("• Was persistency the main driver for Group in Q3?")

show_route = st.sidebar.checkbox("Show interpreted request", value=False)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def number_config(table: pd.DataFrame) -> dict:
    numeric_names = [
        "Overall Variance",
        "Change in mortality (COB1)",
        "Change in lapse (COB2)",
        "Variance Unexplained through COBS",
        "Accrual True Up",
        "LeftOver Persistency",
        "Reconciliation Difference",
        "Overall Movement",
        "Persistency Movement",
    ]
    config = {}
    for col in table.columns:
        if col in numeric_names or any(col.startswith(prefix) for prefix in ["Overall Variance ", "Persistency "]):
            config[col] = st.column_config.NumberColumn(format="%d")
    return config


def display_table(rows: list[dict], title: str | None = None):
    if not rows:
        return
    if title:
        st.markdown(f"#### {title}")
    table = pd.DataFrame(rows)
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config=number_config(table),
    )


question = st.chat_input("Ask about client, treaty, portfolio, age, gender, quarter, comparisons or rankings")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Interpreting the question and analysing verified workbook data..."):
                answer = agent.ask(question)
        except Exception as exc:
            st.error(
                "The agent could not complete the request. If this is a Gemini model/API error, check GEMINI_MODEL and GOOGLE_API_KEY in .streamlit/secrets.toml."
            )
            st.code(str(exc))
        else:
            if show_route and answer.get("route"):
                with st.expander("Interpreted request"):
                    st.json(answer["route"])

            if answer.get("status") != "success":
                st.warning(answer.get("message", "The request could not be completed."))
                assistant_text = answer.get("message", "The request could not be completed.")
            else:
                st.subheader(answer.get("title", "Premium Variance Analysis"))
                query_type = answer.get("query_type")

                if query_type in {"analysis", "overview"}:
                    totals = answer.get("totals", {})
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Overall Variance", totals.get("Overall Variance", 0))
                    c2.metric("Result", totals.get("Result", ""))
                    c3.metric("Treaties", totals.get("Treaties", 0))
                    display_table(answer.get("rows", []), f"Breakdown by {answer.get('group_by', '')}")
                    if answer.get("rows_truncated"):
                        st.info(f"Detailed display is limited to the first {answer.get('row_limit', 50)} groups. Totals use all matching rows.")
                    insight = answer.get("persistency_insight")
                    if insight:
                        st.info(insight.get("statement", ""))

                elif query_type == "comparison":
                    display_table(answer.get("rows", []), "Quarter comparison")
                    if answer.get("detail_rows"):
                        display_table(answer["detail_rows"], "Treaty-level comparison")
                    st.info(answer.get("persistency_comparison_insight", ""))

                elif query_type == "ranking":
                    display_table(answer.get("rows", []), "Ranking")
                    if answer.get("persistency_note"):
                        st.info(answer["persistency_note"])

                st.markdown("### Management commentary")
                st.markdown(answer.get("commentary", ""))
                if answer.get("narration_error"):
                    with st.expander("Narration error details"):
                        st.code(answer["narration_error"])
                assistant_text = answer.get("commentary", "Analysis completed.")

            st.session_state.messages.append({"role": "assistant", "content": assistant_text})
