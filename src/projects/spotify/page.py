"""
Streamlit page renderer for the Spotify popularity prediction project.
"""

from __future__ import annotations

import streamlit as st

from src.projects.spotify.data import AUDIO_FEATURES, TARGET_COL, load_spotify_data
from src.projects.spotify.explainability import (
    compute_mean_absolute_shap,
    compute_spotify_shap_values,
    get_best_tree_model,
)
from src.projects.spotify.features import (
    get_spotify_correlation_matrix,
    split_and_scale_spotify_data,
)
from src.projects.spotify.inference import (
    build_spotify_input_payload,
    load_spotify_production_bundle,
    predict_with_spotify_artifact,
)
from src.projects.spotify.models import (
    build_spotify_metrics_dataframe,
    get_best_spotify_model_name,
    train_spotify_models,
)
from src.projects.spotify.plots import (
    plot_spotify_correlation_heatmap,
    plot_spotify_mean_abs_shap,
    plot_spotify_model_metrics,
    plot_spotify_popularity_distribution,
    plot_spotify_shap_summary,
    plot_target_correlations,
)
from src.ui.components import render_info_card, render_key_value_rows, render_notes_card
from src.ui.layout import render_page_header


def render_spotify_page() -> None:
    """
    Render the full Spotify popularity prediction project page.
    """
    render_page_header(
        title="🎵 Spotify Song Popularity Predictor",
        subtitle=(
            "A regression workflow covering feature analysis, model comparison, "
            "SHAP explainability, and production inference."
        ),
    )

    st.markdown(
        """
        This project now supports **two complementary modes**:

        - **Interactive ML exploration** for regression analysis and explainability
        - **Production-style inference** using saved model artifacts
        """
    )

    df, _ = load_spotify_data()
    X_train, X_test, y_train, y_test, scaler = split_and_scale_spotify_data(df)
    results = train_spotify_models(X_train, X_test, y_train, y_test)

    tabs = st.tabs(
        [
            "📊 Data Overview",
            "🔗 Feature Analysis",
            "🤖 Model Comparison",
            "💡 SHAP Explainability",
            "🎤 Predict Popularity",
            "📦 Production Model Details",
        ]
    )

    with tabs[0]:
        st.header("Dataset Overview")
        st.write(f"**Rows:** {len(df):,}  |  **Columns:** {len(df.columns)}")

        st.subheader("Raw Data (first 100 rows)")
        st.dataframe(df.head(100), use_container_width=True)

        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe().round(2), use_container_width=True)

        st.subheader("Popularity Distribution")
        st.pyplot(plot_spotify_popularity_distribution(df[TARGET_COL]))

    with tabs[1]:
        st.header("Feature Correlation Analysis")

        corr_df = get_spotify_correlation_matrix(df)

        st.subheader("Correlation Heatmap")
        st.pyplot(plot_spotify_correlation_heatmap(corr_df))

        st.subheader("Top Features Correlated with Popularity")
        st.pyplot(plot_target_correlations(corr_df, TARGET_COL))

        st.info(
            "Multicollinearity matters most for linear models. Tree-based models are generally "
            "more robust, but correlation analysis is still useful for understanding the data."
        )

        high_corr_pairs = []
        for i in range(len(AUDIO_FEATURES)):
            for j in range(i + 1, len(AUDIO_FEATURES)):
                r_value = corr_df.iloc[i, j]
                if abs(r_value) > 0.8:
                    high_corr_pairs.append((AUDIO_FEATURES[i], AUDIO_FEATURES[j], round(r_value, 3)))

        if high_corr_pairs:
            st.warning(
                "Highly correlated pairs (|r| > 0.8): "
                + ", ".join(f"{a} ↔ {b} ({r})" for a, b, r in high_corr_pairs)
            )
        else:
            st.success("No feature pairs exceed |r| > 0.8 — multicollinearity is low.")

    with tabs[2]:
        st.header("Model Comparison")
        st.markdown(
            "All models are evaluated on a held-out 20% test set using **MAE**, **RMSE**, and **R²**."
        )

        metrics_df = build_spotify_metrics_dataframe(results)
        st.dataframe(metrics_df, use_container_width=True)
        st.pyplot(plot_spotify_model_metrics(metrics_df))

        best_model_name = metrics_df["R²"].idxmax()
        st.success(
            f"🏆 Best exploration model by R²: {best_model_name} "
            f"(R² = {metrics_df.loc[best_model_name, 'R²']:.4f})"
        )

    with tabs[3]:
        st.header("SHAP Feature Importance")
        st.markdown(
            "This tab remains interactive and educational. It uses the best tree-based exploration model."
        )

        best_tree_name, best_tree_model = get_best_tree_model(results)
        st.write(f"Using **{best_tree_name}** for SHAP analysis.")

        shap_values = compute_spotify_shap_values(best_tree_model, X_test)

        st.subheader("SHAP Summary Plot")
        st.pyplot(plot_spotify_shap_summary(shap_values, X_test, AUDIO_FEATURES))

        st.subheader("Mean |SHAP| Feature Importance")
        mean_abs_shap, feature_names = compute_mean_absolute_shap(shap_values, AUDIO_FEATURES)
        st.pyplot(plot_spotify_mean_abs_shap(mean_abs_shap, feature_names))

    with tabs[4]:
        st.header("Predict a Song's Popularity")
        st.markdown(
            """
            This form uses the **saved production model artifact** instead of retraining at runtime.
            """
        )

        col1, col2 = st.columns(2)

        with col1:
            danceability = st.slider("Danceability", 0.0, 1.0, 0.6, 0.01)
            energy = st.slider("Energy", 0.0, 1.0, 0.7, 0.01)
            valence = st.slider("Valence (positiveness)", 0.0, 1.0, 0.5, 0.01)
            acousticness = st.slider("Acousticness", 0.0, 1.0, 0.2, 0.01)
            speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05, 0.01)
            instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0, 0.01)
            liveness = st.slider("Liveness", 0.0, 1.0, 0.15, 0.01)

        with col2:
            loudness = st.slider("Loudness (dB)", -60.0, 0.0, -6.0, 0.5)
            tempo = st.slider("Tempo (BPM)", 50.0, 200.0, 120.0, 1.0)
            duration_ms = st.slider("Duration (ms)", 60000, 420000, 210000, 1000)
            key = st.slider("Key (0–11)", 0, 11, 5)
            mode = st.selectbox("Mode", [0, 1], format_func=lambda x: "Minor" if x == 0 else "Major")
            time_signature = st.selectbox("Time Signature", [3, 4, 5, 6, 7])
            explicit = st.selectbox("Explicit", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

        if st.button("🎶 Predict Popularity", type="primary"):
            input_payload = build_spotify_input_payload(
                duration_ms=duration_ms,
                explicit=explicit,
                danceability=danceability,
                energy=energy,
                key=key,
                loudness=loudness,
                mode=mode,
                speechiness=speechiness,
                acousticness=acousticness,
                instrumentalness=instrumentalness,
                liveness=liveness,
                valence=valence,
                tempo=tempo,
                time_signature=time_signature,
            )

            try:
                result = predict_with_spotify_artifact(input_payload=input_payload)
                prediction = result["prediction"]

                st.caption(
                    f"Using production model **{result['model_name']}** · Version **{result['version']}**"
                )
                st.metric(label="Predicted Popularity", value=f"{prediction:.1f} / 100")

                if prediction >= 70:
                    st.success("🔥 This song is predicted to be a hit!")
                elif prediction >= 40:
                    st.info("🎵 Moderate popularity expected.")
                else:
                    st.warning("📉 This song may struggle to gain traction.")

            except FileNotFoundError:
                st.error(
                    "No production artifact found yet. Run the offline Spotify training pipeline first."
                )
            except Exception as exc:
                st.error(f"Production inference failed: {exc}")

    with tabs[5]:
        st.subheader("Production Model Details")

        try:
            bundle = load_spotify_production_bundle()
            model_card = bundle["model_card"]
            metrics = bundle["metrics"]
            training_config = bundle["training_config"]

            top1, top2, top3, top4 = st.columns(4)

            with top1:
                render_info_card(
                    title="Artifact Version",
                    value=str(bundle["version"]),
                    subtitle="Current production bundle",
                )

            with top2:
                render_info_card(
                    title="Production Model",
                    value=str(model_card.get("model_name", "N/A")),
                    subtitle="Selected best-fit regressor",
                )

            with top3:
                render_info_card(
                    title="Selection Metric",
                    value=str(training_config.get("selection_metric", "N/A")),
                    subtitle="Metric used for selection",
                )

            with top4:
                created_at = model_card.get("created_at_utc", "N/A")
                render_info_card(
                    title="Trained At",
                    value=created_at[:10] if created_at != "N/A" else "N/A",
                    subtitle="UTC training date",
                )

            st.markdown("### Production Metrics")
            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:
                render_info_card(title="MAE", value=str(metrics.get("MAE", "N/A")))

            with metric_col2:
                render_info_card(title="RMSE", value=str(metrics.get("RMSE", "N/A")))

            with metric_col3:
                render_info_card(title="R²", value=str(metrics.get("R2", "N/A")))

            summary_items = [
                ("Model Name", str(model_card.get("model_name", "N/A"))),
                ("Version", str(model_card.get("version", "N/A"))),
                ("Selection Metric", str(training_config.get("selection_metric", "N/A"))),
                ("Test Size", str(training_config.get("test_size", "N/A"))),
                ("Random State", str(training_config.get("random_state", "N/A"))),
            ]

            st.markdown("### Production Summary")
            render_key_value_rows(summary_items, columns=2)

            notes = model_card.get("notes", "")
            if notes:
                st.markdown("### Notes")
                render_notes_card(
                    title="Model Notes",
                    text=notes,
                )

        except FileNotFoundError:
            st.warning("No production artifact is available yet. Run the Spotify offline training pipeline first.")
        except Exception as exc:
            st.error(f"Failed to load production model details: {exc}")