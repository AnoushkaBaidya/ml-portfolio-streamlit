"""
Streamlit page entry point for the churn project.
"""

import streamlit as st

from src.projects.churn.page import render_churn_page

st.set_page_config(
    page_title="Customer Churn",
    page_icon="📉",
    layout="wide",
)

render_churn_page()