"""
Minimal Streamlit smoke test application.

Purpose:
- Verify that Streamlit is installed correctly
- Verify that the Python virtual environment works
- Verify that the project folder is set up properly
"""

import streamlit as st

# Configure the Streamlit page before rendering any UI.
st.set_page_config(
    page_title="ML Portfolio Setup Test",
    page_icon="✅",
    layout="wide",
)

# Main page content.
st.title("✅ ML Portfolio Environment Setup Successful")
st.write("Your MacBook project environment is ready.")
st.info("Next step: build the professional multi-page project structure.")