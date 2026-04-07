"""
Layout helpers for Streamlit pages.

This module contains small helper functions that keep page files
cleaner and more standardized.
"""

import streamlit as st

from src.ui.theme import apply_app_theme


def render_page_header(title: str, subtitle: str) -> None:
    """
    Render a standardized page header.

    Parameters
    ----------
    title : str
        Main page title.
    subtitle : str
        Supporting one-line explanation under the title.
    """
    apply_app_theme()
    st.title(title)
    st.caption(subtitle)
    st.divider()