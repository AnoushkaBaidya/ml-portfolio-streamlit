"""
Streamlit page renderer for the telecom churn prediction project.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from imblearn.over_sampling import SMOTE

from src.projects.churn.data import RANDOM_STATE, TARGET_COL, load_churn_data
from src.projects.churn.features import engineer_churn_features, preprocess_churn_data
from src.projects.churn.inference import (
    load_churn_production_bundle,
    predict_with_churn_artifact,
)
from src.projects.churn.models import (
    evaluate_churn_models,
    get_churn_feature_importance,
    train_churn_models,
)
from src.projects.churn.plots import (
    plot_before_after_smote,
    plot_churn_distribution,
    plot_churn_rate_by_contract,
    plot_confusion_matrix_heatmap,
    plot_feature_importance,
    plot_precision_recall_curve_with_marker,
    plot_threshold_sweep,
)
from src.projects.churn.thresholding import (
    build_threshold_sweep_dataframe,
    compute_precision_recall_curve_data,
    compute_threshold_metrics,
)
from src.ui.components import render_info_card, render_key_value_rows, render_notes_card
from src.ui.layout import render_page_header


def render_churn_page() -> None:
    """
    Render the full churn project page.
    """
    render_page_header(
        title="📉 Telecom Customer Churn Predictor",
        subtitle=(
            "A classification workflow covering feature engineering, imbalance handling, "
            "threshold tuning, and production inference."
        ),
    )

    st.markdown(
        """
        This project supports **two complementary modes**:

        - **Interactive ML exploration** for threshold tuning and churn analysis
        - **Production-style inference** using saved model artifacts
        """
    )

    raw_df, _ = load_churn_data()
    df = engineer_churn_features(raw_df)
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_churn_data(df)

    tabs = st.tabs(
        [
            "📊 Data Overview",
            "🔧 Feature Engineering",
            "⚖️ Imbalanced Data",
            "🎯 Threshold Tuning",
            "🔮 Predict Churn",
            "📦 Production Model Details",
        ]
    )

    with tabs[0]:
        st.header("Raw Data & Churn Distribution")

        churn_pct = (raw_df[TARGET_COL] == "Yes").mean() * 100
        st.markdown(
            f"The dataset contains **{len(raw_df):,}** customers with "
            f"**{churn_pct:.1f}%** churn rate."
        )

        with st.expander("View raw data sample"):
            st.dataframe(raw_df.head(100), use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Churn Distribution")
            st.pyplot(plot_churn_distribution(raw_df[TARGET_COL]))

        with col2:
            st.subheader("Key Numeric Stats")
            st.dataframe(
                raw_df[["tenure", "MonthlyCharges", "TotalCharges"]].describe().round(2),
                use_container_width=True,
            )

        st.subheader("Churn Rate by Contract Type")
        st.pyplot(plot_churn_rate_by_contract(raw_df, TARGET_COL))

    with tabs[1]:
        st.header("Feature Engineering")
        st.markdown(
            """
            Feature engineering creates new signals that capture churn behavior better than
            raw columns alone.
            """
        )

        with st.expander("View engineered features sample"):
            engineered_cols = [
                "customerID",
                "tenure",
                "tenure_bin",
                "MonthlyCharges",
                "TotalCharges",
                "AvgMonthlyCharge",
                "ChargeRatio",
                "NumServices",
                "HasInternet",
                "IsAutoPayment",
                "Churn",
            ]
            st.dataframe(df[engineered_cols].head(50), use_container_width=True)

        models_plain = train_churn_models(X_train, y_train, use_smote=False)
        random_forest_model = models_plain["Random Forest"]
        feature_importance = get_churn_feature_importance(random_forest_model, feature_names)

        st.subheader("Feature Importance (Random Forest)")
        st.pyplot(plot_feature_importance(feature_importance))

        st.info(
            "Contract type, tenure, and charges tend to dominate churn prediction, "
            "but engineered features like NumServices and ChargeRatio also add value."
        )

    with tabs[2]:
        st.header("The Accuracy Trap & SMOTE")

        st.markdown(
            """
            On imbalanced churn data, a model can get high accuracy by mostly predicting
            the majority class. That is why precision, recall, and F1 matter more.
            """
        )

        majority_acc = round((y_test == 0).mean() * 100, 1)
        st.metric('"Always predict No Churn" accuracy', f"{majority_acc}%")
        st.warning(
            f"That {majority_acc}% accuracy catches 0% of actual churners."
        )

        models_no_smote = train_churn_models(X_train, y_train, use_smote=False)
        models_smote = train_churn_models(X_train, y_train, use_smote=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Without SMOTE")
            results_no_smote = evaluate_churn_models(models_no_smote, X_test, y_test)
            st.dataframe(results_no_smote, use_container_width=True, hide_index=True)

        with col2:
            st.subheader("With SMOTE")
            results_smote = evaluate_churn_models(models_smote, X_test, y_test)
            st.dataframe(results_smote, use_container_width=True, hide_index=True)

        smote = SMOTE(random_state=RANDOM_STATE)
        _, y_resampled = smote.fit_resample(X_train, y_train)

        st.subheader("Class Distribution Before & After SMOTE")
        st.pyplot(plot_before_after_smote(y_train, y_resampled))

    with tabs[3]:
        st.header("Decision Threshold Tuning")

        st.markdown(
            """
            This tab remains interactive and educational.

            The threshold is a business decision layer on top of model probabilities.
            Lower thresholds increase recall, while higher thresholds improve precision.
            """
        )

        models_smote = train_churn_models(X_train, y_train, use_smote=True)
        best_model = models_smote["Gradient Boosting"]
        probabilities = best_model.predict_proba(X_test)[:, 1]

        precisions, recalls, thresholds = compute_precision_recall_curve_data(probabilities, y_test)

        threshold = st.slider(
            "Decision threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.01,
        )

        metric_results = compute_threshold_metrics(probabilities, y_test, threshold)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision", f"{metric_results['precision']:.3f}")
        m2.metric("Recall", f"{metric_results['recall']:.3f}")
        m3.metric("F1 Score", f"{metric_results['f1']:.3f}")
        m4.metric("Accuracy", f"{metric_results['accuracy']:.3f}")

        marker_idx = min(
            pd.Series(thresholds).searchsorted(threshold, side="right"),
            len(precisions) - 1,
        )

        st.subheader("Precision–Recall Curve")
        st.pyplot(
            plot_precision_recall_curve_with_marker(
                recalls=recalls,
                precisions=precisions,
                marker_recall=recalls[marker_idx],
                marker_precision=precisions[marker_idx],
                threshold=threshold,
            )
        )

        sweep_df = build_threshold_sweep_dataframe(probabilities, y_test)

        st.subheader("Metrics Across All Thresholds")
        st.pyplot(plot_threshold_sweep(sweep_df, threshold))

        st.subheader("Confusion Matrix")
        st.pyplot(plot_confusion_matrix_heatmap(metric_results["confusion_matrix"]))

    with tabs[4]:
        st.header("Predict Churn for a Customer")
        st.markdown(
            """
            This form uses the **saved production model artifact** for prediction.
            """
        )

        with st.form("predict_churn_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                senior = st.selectbox("Senior Citizen", [0, 1])
                partner = st.selectbox("Partner", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["Yes", "No"])
                tenure = st.slider("Tenure (months)", 1, 72, 12)
                phone = st.selectbox("Phone Service", ["Yes", "No"])

            with col2:
                multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
                internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
                backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
                protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
                tech = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

            with col3:
                streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
                streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
                payment = st.selectbox(
                    "Payment Method",
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ],
                )
                monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0)
                total = st.number_input("Total Charges ($)", 0.0, 10000.0, monthly * tenure)

            submitted = st.form_submit_button("🔮 Predict Churn")

        if submitted:
            service_values = [security, backup, protection, tech, streaming_tv, streaming_movies]
            num_services = sum(1 for value in service_values if value == "Yes")

            row = pd.DataFrame(
                [
                    {
                        "gender": gender,
                        "SeniorCitizen": senior,
                        "Partner": partner,
                        "Dependents": dependents,
                        "tenure": tenure,
                        "PhoneService": phone,
                        "MultipleLines": multiple_lines,
                        "InternetService": internet,
                        "OnlineSecurity": security,
                        "OnlineBackup": backup,
                        "DeviceProtection": protection,
                        "TechSupport": tech,
                        "StreamingTV": streaming_tv,
                        "StreamingMovies": streaming_movies,
                        "Contract": contract,
                        "PaperlessBilling": paperless,
                        "PaymentMethod": payment,
                        "MonthlyCharges": monthly,
                        "TotalCharges": total,
                        "AvgMonthlyCharge": round(total / max(tenure, 1), 2),
                        "ChargeRatio": round(monthly / (total + 1), 4),
                        "NumServices": num_services,
                        "HasInternet": int(internet != "No"),
                        "IsAutoPayment": int(
                            payment in [
                                "Bank transfer (automatic)",
                                "Credit card (automatic)",
                            ]
                        ),
                    }
                ]
            )

            try:
                result = predict_with_churn_artifact(row_df=row)

                st.caption(
                    f"Using production model **{result['model_name']}** · "
                    f"Version **{result['version']}** · Threshold **{result['threshold']:.2f}**"
                )

                churn_probability = result["probability"]
                label = "⚠️ **Likely to Churn**" if result["prediction"] == 1 else "✅ **Likely to Stay**"

                st.divider()
                st.subheader("Prediction Result")

                result_col1, result_col2 = st.columns(2)
                result_col1.metric("Churn Probability", f"{churn_probability:.1%}")
                result_col2.markdown(f"### {label}")

                if result["prediction"] == 1:
                    st.error(
                        "This customer has elevated churn risk. Consider retention outreach, "
                        "discount offers, or dedicated support."
                    )
                else:
                    st.success(
                        "This customer appears relatively stable. Continue monitoring for changes."
                    )

            except FileNotFoundError:
                st.error(
                    "No production artifact found yet. Run the offline churn training pipeline first."
                )
            except Exception as exc:
                st.error(f"Production inference failed: {exc}")

    with tabs[5]:
        st.subheader("Production Model Details")

        try:
            bundle = load_churn_production_bundle()
            model_card = bundle["model_card"]
            metrics = bundle["metrics"]
            training_config = bundle["training_config"]
            recommended_threshold = model_card.get("hyperparameters", {}).get("recommended_threshold", "N/A")

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
                    subtitle="Selected best-fit classifier",
                )

            with top3:
                render_info_card(
                    title="Selection Metric",
                    value=str(training_config.get("selection_metric", "N/A")),
                    subtitle="Metric used for selection",
                )

            with top4:
                render_info_card(
                    title="Threshold",
                    value=str(recommended_threshold),
                    subtitle="Recommended production threshold",
                )

            st.markdown("### Production Metrics")
            metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

            with metric_col1:
                render_info_card(title="Accuracy", value=str(metrics.get("Accuracy", "N/A")))
            with metric_col2:
                render_info_card(title="Precision", value=str(metrics.get("Precision", "N/A")))
            with metric_col3:
                render_info_card(title="Recall", value=str(metrics.get("Recall", "N/A")))
            with metric_col4:
                render_info_card(title="F1", value=str(metrics.get("F1", "N/A")))
            with metric_col5:
                render_info_card(title="ROC-AUC", value=str(metrics.get("ROC-AUC", "N/A")))

            summary_items = [
                ("Model Name", str(model_card.get("model_name", "N/A"))),
                ("Version", str(model_card.get("version", "N/A"))),
                ("Selection Metric", str(training_config.get("selection_metric", "N/A"))),
                ("Test Size", str(training_config.get("test_size", "N/A"))),
                ("Use SMOTE", str(training_config.get("use_smote_for_production", "N/A"))),
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
            st.warning("No production artifact is available yet. Run the offline churn training pipeline first.")
        except Exception as exc:
            st.error(f"Failed to load production model details: {exc}")