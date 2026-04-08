"""
Streamlit page renderer for the Netflix clustering project.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.projects.netflix.clustering import compute_k_selection_metrics
from src.projects.netflix.data import load_netflix_data
from src.projects.netflix.features import engineer_netflix_features, scale_netflix_features
from src.projects.netflix.inference import (
    assign_clusters_with_artifact,
    load_netflix_production_bundle,
)
from src.projects.netflix.plots import (
    plot_calinski_harabasz_curve,
    plot_davies_bouldin_curve,
    plot_elbow_curve,
    plot_pca_cluster_scatter,
    plot_pca_cluster_scatter_3d,
    plot_silhouette_curve,
    plot_value_distribution,
)
from src.ui.components import render_info_card, render_key_value_rows, render_notes_card
from src.ui.layout import render_page_header


def render_netflix_page() -> None:
    """
    Render the full Netflix clustering project page.
    """
    render_page_header(
        title="🎬 Netflix Show Clustering",
        subtitle=(
            "An unsupervised learning workflow covering data preparation, feature scaling, "
            "KMeans clustering, K selection, PCA visualization, and production artifacts."
        ),
    )

    st.markdown(
        """
        This project now supports **two complementary modes**:

        - **Interactive clustering exploration** for learning and experimentation
        - **Production-style clustering inference** using saved artifacts

        This keeps the project educational while also demonstrating production ML design.
        """
    )

    df, _ = load_netflix_data()

    st.sidebar.header("⚙️ Netflix Settings")
    num_clusters = st.sidebar.slider(
        "Exploration K",
        min_value=2,
        max_value=10,
        value=4,
        key="netflix_num_clusters",
    )

    clean_df, encoded_df, genre_cols = engineer_netflix_features(df)
    scaled_features, feature_names = scale_netflix_features(encoded_df)

    tabs = st.tabs(
        [
            "📊 Data Overview",
            "🔧 Feature Engineering",
            "📐 K Selection",
            "🗺️ 2D Cluster Visualisation",
            "🧊 3D Cluster Visualisation",
            "🔍 Cluster Analysis",
            "📦 Production Model Details",
        ]
    )

    with tabs[0]:
        st.subheader("Dataset at a Glance")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Titles", len(df))

        with col2:
            st.metric("Movies", int((df["type"] == "Movie").sum()))

        with col3:
            st.metric("TV Shows", int((df["type"] == "TV Show").sum()))

        with st.expander("Show raw data"):
            st.dataframe(df, use_container_width=True)

        with st.expander("Descriptive statistics"):
            st.dataframe(df.describe(include="all").T.fillna(""), use_container_width=True)

        st.subheader("Distributions")

        dcol1, dcol2 = st.columns(2)

        with dcol1:
            st.pyplot(plot_value_distribution(df["type"], "Content Type"))
            st.pyplot(plot_value_distribution(df["rating"].fillna("NR"), "Ratings"))

        with dcol2:
            genre_series = df["listed_in"].fillna("").str.split(", ").explode()
            st.pyplot(plot_value_distribution(genre_series, "Top Genres", top_n=12))
            st.pyplot(plot_value_distribution(df["release_year"].astype(str), "Release Year", top_n=10))

    with tabs[1]:
        st.subheader("Turning Categories into Numbers")

        st.markdown("#### 1 · Label Encoding the Rating Column")
        st.markdown(
            """
            **Label encoding** assigns an integer to each content rating category so it can be
            used numerically in clustering.
            """
        )

        sample_rating = clean_df[["title", "rating", "rating_encoded"]].drop_duplicates(subset="rating").head(8)
        st.dataframe(sample_rating, use_container_width=True)

        st.markdown("#### 2 · One-Hot Encoding Genres")
        st.markdown(
            """
            Each title may belong to multiple genres. One-hot encoding captures genre membership
            without introducing false ordinal meaning.
            """
        )

        st.dataframe(
            clean_df[["title", "listed_in"] + genre_cols].head(8),
            use_container_width=True,
        )

        st.markdown("#### 3 · Feature Scaling")
        st.markdown(
            """
            KMeans uses Euclidean distance, so features must be scaled to comparable ranges.
            Without scaling, large-range variables dominate cluster assignment.
            """
        )

        before_after_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Mean (raw)": encoded_df.mean().round(2).values,
                "Std (raw)": encoded_df.std().round(2).values,
                "Mean (scaled)": scaled_features.mean(axis=0).round(4),
                "Std (scaled)": scaled_features.std(axis=0).round(4),
            }
        )
        st.dataframe(before_after_df, use_container_width=True)

        st.markdown("#### 4 · Final Feature Matrix Preview")
        st.dataframe(encoded_df.head(10), use_container_width=True)

    with tabs[2]:
        st.subheader("Hyperparameter Search for K")
        st.markdown(
            """
            In clustering, **K** is the core hyperparameter.

            A production-style workflow should not pick K from a single chart alone.
            Here we evaluate multiple candidate K values across several metrics.
            """
        )

        metrics_df = compute_k_selection_metrics(
            scaled_features=scaled_features,
            k_values=tuple(range(2, 11)),
        )

        top_left, top_right = st.columns(2)
        with top_left:
            st.pyplot(plot_elbow_curve(metrics_df))
            st.pyplot(plot_davies_bouldin_curve(metrics_df))

        with top_right:
            st.pyplot(plot_silhouette_curve(metrics_df))
            st.pyplot(plot_calinski_harabasz_curve(metrics_df))

        st.dataframe(metrics_df.set_index("K").round(4), use_container_width=True)

        best_k = int(metrics_df.sort_values(by="Silhouette", ascending=False).iloc[0]["K"])
        st.success(f"Best exploration K by Silhouette Score: **{best_k}**")

    with tabs[3]:
        st.subheader("2D PCA Cluster Visualisation")

        try:
            result = assign_clusters_with_artifact(encoded_df=encoded_df)
            st.caption(
                f"Using production clustering artifact · Model **{result['model_name']}** · Version **{result['version']}**"
            )
            st.pyplot(plot_pca_cluster_scatter(result["pca_2d"], result["labels"]))
        except FileNotFoundError:
            st.warning("No production artifact found yet. Run the offline Netflix training pipeline first.")
        except Exception as exc:
            st.error(f"Failed to load production 2D clustering view: {exc}")

    with tabs[4]:
        st.subheader("3D PCA Cluster Visualisation")
        st.markdown(
            """
            This 3D PCA view gives a richer visual sense of cluster separation than the 2D projection.
            """
        )

        try:
            result = assign_clusters_with_artifact(encoded_df=encoded_df)
            st.caption(
                f"Using production clustering artifact · Model **{result['model_name']}** · Version **{result['version']}**"
            )
            st.pyplot(plot_pca_cluster_scatter_3d(result["pca_3d"], result["labels"]))
        except FileNotFoundError:
            st.warning("No production artifact found yet. Run the offline Netflix training pipeline first.")
        except Exception as exc:
            st.error(f"Failed to load production 3D clustering view: {exc}")

    with tabs[5]:
        st.subheader("Cluster Analysis")

        try:
            result = assign_clusters_with_artifact(encoded_df=encoded_df)
            labels = result["labels"]

            analysis_df = clean_df.copy()
            analysis_df["Cluster"] = labels

            unique_clusters = sorted(analysis_df["Cluster"].unique())

            for cluster_id in unique_clusters:
                subset = analysis_df[analysis_df["Cluster"] == cluster_id]

                st.markdown(f"---\n### Cluster {cluster_id} ({len(subset)} titles)")

                m1, m2, m3 = st.columns(3)

                with m1:
                    st.metric("Titles", len(subset))

                with m2:
                    st.metric("Avg Release Year", int(subset["release_year"].mean()))

                with m3:
                    movie_ratio = (subset["type"] == "Movie").mean()
                    st.metric("% Movies", f"{movie_ratio:.0%}")

                info1, info2 = st.columns(2)

                with info1:
                    st.markdown("**Top Ratings**")
                    top_ratings = subset["rating"].value_counts().head(5).reset_index()
                    st.dataframe(top_ratings, use_container_width=True)

                with info2:
                    st.markdown("**Top Genres**")
                    top_genres = (
                        subset["listed_in"]
                        .fillna("")
                        .str.split(", ")
                        .explode()
                        .value_counts()
                        .head(5)
                        .reset_index()
                    )
                    st.dataframe(top_genres, use_container_width=True)

                with st.expander(f"Sample titles from Cluster {cluster_id}"):
                    st.dataframe(
                        subset[["title", "type", "rating", "release_year", "listed_in"]].head(10),
                        use_container_width=True,
                    )

        except FileNotFoundError:
            st.warning("No production artifact found yet. Run the offline Netflix training pipeline first.")
        except Exception as exc:
            st.error(f"Failed to load production cluster analysis: {exc}")

    with tabs[6]:
        st.subheader("Production Model Details")

        try:
            bundle = load_netflix_production_bundle()
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
                    subtitle="Selected clustering model",
                )

            with top3:
                render_info_card(
                    title="Best K",
                    value=str(metrics.get("Best K", "N/A")),
                    subtitle="Production cluster count",
                )

            with top4:
                created_at = model_card.get("created_at_utc", "N/A")
                render_info_card(
                    title="Trained At",
                    value=created_at[:10] if created_at != "N/A" else "N/A",
                    subtitle="UTC training date",
                )

            st.markdown("### Production Metrics")
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

            with metric_col1:
                render_info_card(
                    title="Silhouette",
                    value=str(metrics.get("Silhouette", "N/A")),
                )

            with metric_col2:
                render_info_card(
                    title="Davies-Bouldin",
                    value=str(metrics.get("Davies-Bouldin", "N/A")),
                )

            with metric_col3:
                render_info_card(
                    title="Calinski-Harabasz",
                    value=str(metrics.get("Calinski-Harabasz", "N/A")),
                )

            with metric_col4:
                render_info_card(
                    title="Inertia",
                    value=str(metrics.get("Inertia", "N/A")),
                )

            summary_items = [
                ("Model Name", str(model_card.get("model_name", "N/A"))),
                ("Version", str(model_card.get("version", "N/A"))),
                ("Selection Metric", str(training_config.get("selection_metric", "N/A"))),
                ("Candidate K Values", str(training_config.get("candidate_k_values", "N/A"))),
                ("Random State", str(training_config.get("random_state", "N/A"))),
            ]

            st.markdown("### Production Summary")
            render_key_value_rows(summary_items, columns=2)

            hyperparameter_items = [
                (str(key), str(value))
                for key, value in model_card.get("hyperparameters", {}).items()
            ]

            if hyperparameter_items:
                st.markdown("### Selected Hyperparameters")
                render_key_value_rows(hyperparameter_items, columns=2)

            notes = model_card.get("notes", "")
            if notes:
                st.markdown("### Notes")
                render_notes_card(
                    title="Model Notes",
                    text=notes,
                )

        except FileNotFoundError:
            st.warning("No production artifact is available yet. Run the Netflix offline training pipeline first.")
        except Exception as exc:
            st.error(f"Failed to load production model details: {exc}")