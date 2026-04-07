"""
Streamlit page entry point for the diabetes prediction project.
"""

import streamlit as st

from src.projects.diabetes.page import render_diabetes_page

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide",
)

render_diabetes_page()