"""
Streamlit page entry point for the Netflix clustering project.
"""

import streamlit as st

from src.projects.netflix.page import render_netflix_page

st.set_page_config(
    page_title="Netflix Clustering",
    page_icon="🎬",
    layout="wide",
)

render_netflix_page()