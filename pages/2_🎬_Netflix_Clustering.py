import streamlit as st
from src.ui.layout import render_page_header

st.set_page_config(page_title="Netflix Clustering", page_icon="🎬", layout="wide")

render_page_header(
    title="🎬 Netflix Clustering",
    subtitle="This page will contain the modularized Netflix clustering project.",
)

st.info("Placeholder page created successfully. Next, we will migrate the real Netflix clustering app here.")