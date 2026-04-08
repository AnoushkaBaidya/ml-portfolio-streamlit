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


def test_import_diabetes_inference():
    from src.projects.diabetes.inference import predict_with_diabetes_artifact

    assert callable(predict_with_diabetes_artifact)


def test_import_diabetes_training():
    from src.projects.diabetes.train import run_diabetes_training_pipeline

    assert callable(run_diabetes_training_pipeline)


def test_import_netflix_page():
    from src.projects.netflix.page import render_netflix_page

    assert callable(render_netflix_page)


def test_import_netflix_loader():
    from src.projects.netflix.data import load_netflix_data

    assert callable(load_netflix_data)


def test_import_netflix_inference():
    from src.projects.netflix.inference import assign_clusters_with_artifact

    assert callable(assign_clusters_with_artifact)


def test_import_netflix_training():
    from src.projects.netflix.train import run_netflix_training_pipeline

    assert callable(run_netflix_training_pipeline)


def test_import_spotify_page():
    from src.projects.spotify.page import render_spotify_page

    assert callable(render_spotify_page)


def test_import_spotify_loader():
    from src.projects.spotify.data import load_spotify_data

    assert callable(load_spotify_data)


def test_import_churn_page():
    from src.projects.churn.page import render_churn_page

    assert callable(render_churn_page)


def test_import_churn_loader():
    from src.projects.churn.data import load_churn_data

    assert callable(load_churn_data)