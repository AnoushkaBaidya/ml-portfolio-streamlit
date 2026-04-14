# ML Portfolio Streamlit App

A multi-page machine learning portfolio application built with Streamlit.

## Live Demo

[Deployed Streamlit link here after deployment.]

## Overview

This portfolio combines four interactive machine learning projects into one friendly web application.

It is designed to show both:

- interactive ML exploration
- production-style inference using saved artifacts

## Included Projects

### 1. Diabetes Prediction
Binary classification project demonstrating:
- feature scaling
- model comparison
- hyperparameter tuning
- saved production model inference

### 2. Netflix Clustering
Unsupervised learning project demonstrating:
- categorical feature engineering
- KMeans clustering
- K selection with multiple metrics
- PCA-based 2D and 3D visualization
- saved production clustering artifacts

### 3. Spotify Popularity Prediction
Regression project demonstrating:
- feature correlation analysis
- model benchmarking
- SHAP explainability
- saved production model inference

### 4. Customer Churn Prediction
Classification project demonstrating:
- feature engineering
- imbalance handling with SMOTE
- threshold tuning
- saved production scoring

## Architecture

This project uses a hybrid architecture:

### Interactive exploration
Used for:
- learning
- experimentation
- hyperparameter exploration
- metric visualization

### Production inference
Used for:
- saved model loading
- artifact-backed predictions
- versioned model details
- reproducible deployment behavior

## Artifact Structure

```text
artifacts/
└── processed/
    ├── diabetes/
    ├── netflix/
    ├── spotify/
    └── churn/