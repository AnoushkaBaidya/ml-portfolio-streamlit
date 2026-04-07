
"""
Basic smoke tests to confirm that core modules import successfully.

These tests are intentionally minimal at this stage. Their purpose is
to catch package or path issues early before project logic is added.
"""


def test_import_config():
    from src import config

    assert config.APP_NAME == "ML Portfolio Streamlit"


def test_import_theme():
    from src.ui.theme import apply_app_theme

    assert callable(apply_app_theme)