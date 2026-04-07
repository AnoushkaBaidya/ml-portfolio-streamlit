"""
Basic smoke tests to confirm that core modules import successfully.
"""


def test_import_config():
    from src import config

    assert config.APP_NAME == "ML Portfolio Streamlit"


def test_import_theme():
    from src.ui.theme import apply_app_theme

    assert callable(apply_app_theme)


def test_import_diabetes_page():
    from src.projects.diabetes.page import render_diabetes_page

    assert callable(render_diabetes_page)


def test_import_diabetes_loader():
    from src.projects.diabetes.data import load_diabetes_data

    assert callable(load_diabetes_data)


def test_import_netflix_page():
    from src.projects.netflix.page import render_netflix_page

    assert callable(render_netflix_page)


def test_import_netflix_loader():
    from src.projects.netflix.data import load_netflix_data

    assert callable(load_netflix_data)