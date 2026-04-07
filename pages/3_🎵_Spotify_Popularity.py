import streamlit as st
from src.ui.layout import render_page_header

st.set_page_config(page_title="Spotify Popularity", page_icon="🎵", layout="wide")

render_page_header(
    title="🎵 Spotify Popularity Prediction",
    subtitle="This page will contain the modularized Spotify regression project.",
)

st.info("Placeholder page created successfully. Next, we will migrate the real Spotify app here.")