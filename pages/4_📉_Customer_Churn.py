import streamlit as st
from src.ui.layout import render_page_header

st.set_page_config(page_title="Customer Churn", page_icon="📉", layout="wide")

render_page_header(
    title="📉 Customer Churn Prediction",
    subtitle="This page will contain the modularized telecom churn project.",
)

st.info("Placeholder page created successfully. Next, we will migrate the real churn app here.")