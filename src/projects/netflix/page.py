"""
Streamlit page renderer for the Netflix clustering project.
"""

from __future__ import annotations

import streamlit as st
from sklearn.metrics import silhouette_score
import pandas as pd

from src.projects.netflix.clustering import (
    compute_k_selection_metrics,
    run_netflix_kmeans,
    run_netflix_pca,
)
from src.projects.netflix.data import load_netflix_data
from src.projects.netflix.features import engineer_netflix_features, scale_netflix_features
from src.projects.netflix.plots import (
    plot_elbow_curve,
    plot_pca_cluster_scatter,
    plot_silhouette_curve,
    plot_value_distribution,
)
from src.ui.layout import render_page_header


def render_netflix_page() -> None:
    """
    Render the full Netflix clustering project page.
    """
    render_page_header(
        title="🎬 Netflix Show Clustering",
        subtitle=(
            "An unsupervised learning workflow covering data preparation, feature scaling, "
            "KMeans clustering, optimal K selection, PCA visualization, and cluster profiling."
        ),
    )

    st.markdown(
        """
        This interactive project demonstrates a complete clustering pipeline
        using Netflix-style catalog data.

        **You will explore:**
        - how categorical media data is turned into numeric features
        - why scaling is critical for KMeans
        - how the elbow method and silhouette score help choose K
        - how PCA helps visualize high-dimensional clusters
        - how to interpret what each cluster represents
        """
    )

    df, data_source_label = load_netflix_data()
    st.info(data_source_label)

    st.sidebar.header("⚙️ Netflix Settings")
    num_clusters = st.sidebar.slider(
        "Number of clusters (K)",
        min_value=2,
        max_value=10,
        value=4,
        key="netflix_num_clusters",
    )

    clean_df, encoded_df, genre_cols = engineer_netflix_features(df)
    scaled_features, feature_names = scale_netflix_features(encoded_df)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Data Overview",
            "🔧 Feature Engineering",
            "📐 Optimal K Selection",
            "🗺️ Cluster Visualisation",
            "🔍 Cluster Analysis",
        ]
    )

    with tab1:
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

    with tab2:
        st.subheader("Turning Categories into Numbers")

        st.markdown("#### 1 · Label Encoding the Rating Column")
        st.markdown(
            """
            **Label encoding** assigns a unique integer to each rating category.
            This creates a compact numeric representation for clustering.
            """
        )

        sample_rating = clean_df[["title", "rating", "rating_encoded"]].drop_duplicates(subset="rating").head(8)
        st.dataframe(sample_rating, use_container_width=True)

        st.markdown("#### 2 · One-Hot Encoding Genres")
        st.markdown(
            """
            Each title can belong to multiple genres. One-hot encoding creates a
            binary column per genre, allowing KMeans to use genre membership as part
            of the clustering space.
            """
        )

        st.dataframe(
            clean_df[["title", "listed_in"] + genre_cols].head(8),
            use_container_width=True,
        )

        st.markdown("#### 3 · Feature Scaling (StandardScaler)")
        st.markdown(
            """
            KMeans uses Euclidean distance. If one feature has a much larger numeric
            range than another, it will dominate the clustering result.

            StandardScaler brings all features onto a comparable scale.
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
        preview_df = encoded_df.copy()
        st.dataframe(preview_df.head(10), use_container_width=True)

    with tab3:
        st.subheader("How Many Clusters?")

        st.markdown(
            """
            Two methods help choose **K**:

            - **Elbow Method** — looks for diminishing returns in inertia
            - **Silhouette Score** — evaluates how well-separated the clusters are
            """
        )

        metrics_df = compute_k_selection_metrics(scaled_features)

        col_elbow, col_silhouette = st.columns(2)

        with col_elbow:
            st.pyplot(plot_elbow_curve(metrics_df))

        with col_silhouette:
            st.pyplot(plot_silhouette_curve(metrics_df))

        st.dataframe(metrics_df.set_index("K").round(4), use_container_width=True)

        best_k = int(metrics_df.loc[metrics_df["Silhouette"].idxmax(), "K"])
        st.info(f"Highest silhouette score occurs at **K = {best_k}**.")

    with tab4:
        st.subheader(f"PCA Scatter Plot (K = {num_clusters})")
        st.markdown(
            """
            PCA projects the high-dimensional feature space into 2 dimensions so
            the clusters can be visualized on a scatter plot.
            """
        )

        labels = run_netflix_kmeans(scaled_features, num_clusters)
        pca_result = run_netflix_pca(scaled_features)

        st.pyplot(plot_pca_cluster_scatter(pca_result, labels))

        score = silhouette_score(scaled_features, labels)
        st.metric("Silhouette Score", f"{score:.4f}")

    with tab5:
        st.subheader(f"What’s Inside Each Cluster? (K = {num_clusters})")
        st.markdown(
            "Below we profile each cluster by dominant content type, ratings, genres, and release years."
        )

        labels = run_netflix_kmeans(scaled_features, num_clusters)
        analysis_df = clean_df.copy()
        analysis_df["Cluster"] = labels

        for cluster_id in sorted(analysis_df["Cluster"].unique()):
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