import streamlit as st
from src.ui.layout import render_page_header

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="wide")

render_page_header(
    title="🩺 Diabetes Prediction",
    subtitle="This page will contain the modularized diabetes classification project.",
)

st.info("Placeholder page created successfully. Next, we will migrate the real diabetes app here.")