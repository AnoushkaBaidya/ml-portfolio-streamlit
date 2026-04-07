"""
Streamlit page renderer for the diabetes prediction project.

This page preserves the interactive educational workflow while also
adding production-style artifact-backed inference and model details.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.projects.diabetes.data import FEATURE_COLS, load_diabetes_data
from src.projects.diabetes.inference import (
    build_diabetes_input_payload,
    load_diabetes_production_bundle,
    predict_with_diabetes_artifact,
)
from src.projects.diabetes.models import (
    compare_diabetes_models,
    evaluate_scaling_strategies,
    fit_models_for_visuals,
    get_diabetes_models,
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
from src.ui.components import (
    render_info_card,
    render_key_value_rows,
    render_notes_card,
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
            "feature scaling, model benchmarking, hyperparameter tuning, and production inference."
        ),
    )

    st.markdown(
        """
        This project now supports **two complementary modes**:

        - **Interactive ML exploration** for learning and experimentation
        - **Production-style inference** using saved model artifacts

        This keeps the app educational while also demonstrating real ML deployment discipline.
        """
    )

    df, _ = load_diabetes_data()
    

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

    tabs = st.tabs(
        [
            "📊 Data Overview",
            "⚖️ Feature Scaling",
            "🤖 Model Comparison",
            "🔧 Hyperparameter Tuning",
            "🔮 Make a Prediction",
            "📦 Production Model Details",
        ]
    )

    with tabs[0]:
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

    with tabs[1]:
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

    with tabs[2]:
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

    with tabs[3]:
        st.subheader("Hyperparameter Tuning with GridSearchCV")
        st.markdown(
            """
            This tab remains interactive and educational.

            It lets you explore how different model families perform when tuned,
            but it does **not** define the production inference artifact directly.
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

    with tabs[4]:
        st.subheader("Interactive Prediction Form")
        st.markdown(
            """
            This form uses the **saved production model artifact** instead of
            retraining at runtime.

            The UI remains interactive, but the prediction path is now production-style.
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
            input_payload = build_diabetes_input_payload(
                pregnancies=pregnancies,
                glucose=glucose,
                blood_pressure=blood_pressure,
                skin_thickness=skin_thickness,
                insulin=insulin,
                bmi=bmi,
                dpf=dpf,
                age=age,
            )

            try:
                result = predict_with_diabetes_artifact(input_payload=input_payload)
                prediction = result["prediction"]
                probabilities = result["probabilities"]

                st.divider()
                st.caption(
                    f"Using production model **{result['model_name']}** · Version **{result['version']}**"
                )

                if prediction == 1:
                    st.error(f"⚠️ **Diabetic** — model confidence: {probabilities[1]:.1%}")
                else:
                    st.success(f"✅ **Non-diabetic** — model confidence: {probabilities[0]:.1%}")

            except FileNotFoundError:
                st.error(
                    "No production artifact found yet. Run the offline diabetes training pipeline first."
                )
            except Exception as exc:
                st.error(f"Production inference failed: {exc}")

    with tabs[5]:
        st.subheader("Production Model Details")

        try:
            bundle = load_diabetes_production_bundle()
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
                    subtitle="Selected best-fit model",
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

            st.markdown("### Validation Metrics")
            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:
                render_info_card(
                    title="Accuracy",
                    value=str(metrics.get("Accuracy", "N/A")),
                )

            with metric_col2:
                render_info_card(
                    title="F1 Score",
                    value=str(metrics.get("F1", "N/A")),
                )

            with metric_col3:
                render_info_card(
                    title="ROC-AUC",
                    value=str(metrics.get("ROC-AUC", "N/A")),
                )

            summary_items = [
                ("Model Name", str(model_card.get("model_name", "N/A"))),
                ("Version", str(model_card.get("version", "N/A"))),
                ("Selection Metric", str(training_config.get("selection_metric", "N/A"))),
                ("Test Size", str(training_config.get("test_size", "N/A"))),
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
            st.warning(
                "No production artifact is available yet. Run the diabetes offline training pipeline first."
            )
        except Exception as exc:
            st.error(f"Failed to load production model details: {exc}")