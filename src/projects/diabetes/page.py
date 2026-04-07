"""
Streamlit page renderer for the diabetes prediction project.

This module is responsible only for rendering the UI. Model logic,
plotting helpers, and dataset loading live in separate modules.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from src.projects.diabetes.data import FEATURE_COLS, load_diabetes_data
from src.projects.diabetes.models import (
    compare_diabetes_models,
    evaluate_scaling_strategies,
    fit_models_for_visuals,
    get_diabetes_models,
    predict_diabetes_outcome,
    run_diabetes_grid_search,
    split_diabetes_data,
)
from src.projects.diabetes.plots import (
    plot_confusion_matrix,
    plot_diabetes_correlation_heatmap,
    plot_model_comparison,
    plot_roc_curves,
    plot_scaling_comparison,
)
from src.ui.layout import render_page_header


def render_diabetes_page() -> None:
    """
    Render the full diabetes project page.
    """
    render_page_header(
        title="🩺 Diabetes Prediction – Model Comparison",
        subtitle=(
            "A complete classification workflow covering data exploration, "
            "feature scaling, model benchmarking, hyperparameter tuning, and live prediction."
        ),
    )

    st.markdown(
        """
        This interactive project demonstrates a complete supervised learning pipeline
        using a diabetes prediction use case.

        **You will explore:**
        - how feature scaling changes model performance
        - how multiple classification models compare
        - how GridSearchCV improves a selected model
        - how to interpret evaluation metrics like F1 and ROC-AUC
        - how to generate a live patient-level prediction
        """
    )

    df, data_source_label = load_diabetes_data()
    st.info(data_source_label)

    st.sidebar.header("⚙️ Diabetes Settings")
    test_size = st.sidebar.slider(
        "Test set size",
        min_value=0.10,
        max_value=0.50,
        value=0.20,
        step=0.05,
        key="diabetes_test_size",
    )
    random_state = st.sidebar.number_input(
        "Random state",
        min_value=0,
        max_value=999,
        value=42,
        key="diabetes_random_state",
    )

    X_train, X_test, y_train, y_test = split_diabetes_data(
        df=df,
        test_size=float(test_size),
        random_state=int(random_state),
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Data Overview",
            "⚖️ Feature Scaling",
            "🤖 Model Comparison",
            "🔧 Hyperparameter Tuning",
            "🔮 Make a Prediction",
        ]
    )

    with tab1:
        st.subheader("Dataset at a Glance")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Rows", df.shape[0])
            st.metric("Features", len(FEATURE_COLS))

        with col2:
            target_counts = df["Outcome"].value_counts()
            st.metric("Non-diabetic (0)", int(target_counts.get(0, 0)))
            st.metric("Diabetic (1)", int(target_counts.get(1, 0)))

        with st.expander("Show raw data"):
            st.dataframe(df, use_container_width=True)

        with st.expander("Descriptive statistics"):
            st.dataframe(df.describe().round(2), use_container_width=True)

        st.subheader("Feature Correlations")
        st.pyplot(plot_diabetes_correlation_heatmap(df))

    with tab2:
        st.subheader("How Feature Scaling Affects Logistic Regression")
        st.markdown(
            """
            Many machine learning algorithms are sensitive to the scale of input features.
            Here we compare three strategies:

            - **No Scaling**
            - **StandardScaler**
            - **MinMaxScaler**
            """
        )

        scaling_df = evaluate_scaling_strategies(X_train, X_test, y_train, y_test)
        st.dataframe(scaling_df.set_index("Scaling"), use_container_width=True)
        st.pyplot(plot_scaling_comparison(scaling_df))

    with tab3:
        st.subheader("Side-by-Side Model Comparison")
        st.markdown(
            "All models below are trained on standardized features and evaluated on the held-out test set."
        )

        comparison_df = compare_diabetes_models(X_train, X_test, y_train, y_test)
        st.dataframe(comparison_df.set_index("Model"), use_container_width=True)
        st.pyplot(plot_model_comparison(comparison_df))

        st.subheader("Confusion Matrices")

        _, X_test_scaled, fitted_models = fit_models_for_visuals(X_train, X_test, y_train)
        model_columns = st.columns(len(fitted_models))

        for idx, (model_name, model) in enumerate(fitted_models.items()):
            predictions = model.predict(X_test_scaled)
            with model_columns[idx]:
                st.pyplot(plot_confusion_matrix(y_test, predictions, title=model_name))

        st.subheader("ROC Curves")
        st.pyplot(plot_roc_curves(X_test_scaled, y_test, fitted_models))

    with tab4:
        st.subheader("Hyperparameter Tuning with GridSearchCV")
        st.markdown(
            """
            Select a model below and run a 5-fold cross-validated grid search
            optimized for ROC-AUC.
            """
        )

        model_choice = st.selectbox(
            "Choose a model to tune",
            list(get_diabetes_models().keys()),
            key="diabetes_model_choice",
        )

        if st.button("Run Grid Search", key="diabetes_grid_search_button"):
            with st.spinner("Searching hyperparameter space..."):
                grid_search_results = run_diabetes_grid_search(
                    X_train=X_train,
                    y_train=y_train,
                    model_name=model_choice,
                    test_size=float(test_size),
                    random_state=int(random_state),
                )

            st.success(
                f"Best CV ROC-AUC: {grid_search_results['best_score']} | "
                f"Best params: {grid_search_results['best_params']}"
            )
            st.dataframe(grid_search_results["cv_results"], use_container_width=True)

    with tab5:
        st.subheader("Enter Patient Features")
        st.markdown(
            """
            Adjust the sliders below and click **Predict**.
            The prediction uses a Random Forest model trained on the training set.
            """
        )

        col1, col2 = st.columns(2)

        with col1:
            pregnancies = st.slider("Pregnancies", 0, 17, 1)
            glucose = st.slider("Glucose (mg/dL)", 0, 200, 120)
            blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 122, 70)
            skin_thickness = st.slider("Skin Thickness (mm)", 0, 99, 20)

        with col2:
            insulin = st.slider("Insulin (μU/mL)", 0, 846, 80)
            bmi = st.slider("BMI", 0.0, 67.0, 32.0, step=0.1)
            dpf = st.slider("Diabetes Pedigree Function", 0.078, 2.42, 0.47, step=0.01)
            age = st.slider("Age", 21, 81, 33)

        if st.button("Predict", key="diabetes_predict_button"):
            patient_features = np.array(
                [[
                    pregnancies,
                    glucose,
                    blood_pressure,
                    skin_thickness,
                    insulin,
                    bmi,
                    dpf,
                    age,
                ]]
            )

            result = predict_diabetes_outcome(
                X_train=X_train,
                y_train=y_train,
                patient_features=patient_features,
            )

            prediction = result["prediction"]
            probabilities = result["probabilities"]

            st.divider()

            if prediction == 1:
                st.error(f"⚠️ **Diabetic** — model confidence: {probabilities[1]:.1%}")
            else:
                st.success(f"✅ **Non-diabetic** — model confidence: {probabilities[0]:.1%}")