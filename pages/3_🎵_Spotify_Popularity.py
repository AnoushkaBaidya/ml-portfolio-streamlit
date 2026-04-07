"""
Streamlit page entry point for the Spotify popularity project.
"""

import streamlit as st

from src.projects.spotify.page import render_spotify_page

st.set_page_config(
    page_title="Spotify Popularity",
    page_icon="🎵",
    layout="wide",
)

render_spotify_page()