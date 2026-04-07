"""
Streamlit page renderer for the telecom churn prediction project.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from imblearn.over_sampling import SMOTE

from src.projects.churn.data import RANDOM_STATE, TARGET_COL, load_churn_data
from src.projects.churn.features import engineer_churn_features, preprocess_churn_data
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
from src.ui.layout import render_page_header


def render_churn_page() -> None:
    """
    Render the full churn project page.
    """
    render_page_header(
        title="📉 Telecom Customer Churn Predictor",
        subtitle=(
            "A classification workflow covering feature engineering, imbalance handling, "
            "threshold tuning, and live churn prediction."
        ),
    )

    st.markdown(
        """
        This interactive project demonstrates a complete churn-prediction workflow.

        **You will explore:**
        - raw telecom data and churn imbalance
        - domain-driven feature engineering
        - why accuracy alone is misleading
        - how SMOTE improves minority-class learning
        - how threshold tuning changes business trade-offs
        - live churn prediction for an individual customer
        """
    )

    raw_df, data_source_label = load_churn_data()
    st.info(data_source_label)

    df = engineer_churn_features(raw_df)
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_churn_data(df)

    tabs = st.tabs(
        [
            "📊 Data Overview",
            "🔧 Feature Engineering",
            "⚖️ Imbalanced Data",
            "🎯 Threshold Tuning",
            "🔮 Predict Churn",
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
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        st.subheader("Class Distribution Before & After SMOTE")
        st.pyplot(plot_before_after_smote(y_train, y_resampled))

    with tabs[3]:
        st.header("Decision Threshold Tuning")

        st.markdown(
            """
            Most classifiers output probabilities, not just labels.

            Lower thresholds increase recall and catch more churners, but also increase
            false positives. Higher thresholds improve precision, but miss more true churners.
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
        st.markdown("Fill in the customer details below and click **Predict**.")

        models_smote = train_churn_models(X_train, y_train, use_smote=True)
        production_model = models_smote["Gradient Boosting"]

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

            row_encoded = pd.get_dummies(row, drop_first=True)
            row_encoded = row_encoded.reindex(columns=feature_names, fill_value=0)
            row_scaled = pd.DataFrame(
                scaler.transform(row_encoded),
                columns=feature_names,
            )

            churn_probability = production_model.predict_proba(row_scaled)[0][1]
            label = "⚠️ **Likely to Churn**" if churn_probability >= 0.5 else "✅ **Likely to Stay**"

            st.divider()
            st.subheader("Prediction Result")

            result_col1, result_col2 = st.columns(2)
            result_col1.metric("Churn Probability", f"{churn_probability:.1%}")
            result_col2.markdown(f"### {label}")

            if churn_probability >= 0.5:
                st.error(
                    "This customer has elevated churn risk. Consider retention outreach, "
                    "discount offers, or dedicated support."
                )
            else:
                st.success(
                    "This customer appears relatively stable. Continue monitoring for changes."
                )