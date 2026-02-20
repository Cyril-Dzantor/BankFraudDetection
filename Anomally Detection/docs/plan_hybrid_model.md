
# Implementation Plan: Hybrid Fraud Detection Architecture (Ensemble)

## 1. Goal
To build a **Hybrid Fraud Detection System** where:
1.  **Stage 1 (Unsupervised)**: Isolation Forest and Lightweight Autoencoder act as feature extractors, generating **Anomaly Scores**.
2.  **Aggregation**: The scores from Stage 1 are combined (e.g., averaged) to create a robust **Ensemble Anomaly Score**.
3.  **Stage 2 (Supervised)**: An **XGBoost Classifier** uses the original transaction features PLUS the Ensemble Anomaly Score to predict the final **Fraud Probability**.

## 2. Directory Structure
Work within the existing `Anomally Detection` folder.
- **Script**: `h:\Axxend Project\BankFraudDetection\Anomally Detection\train_hybrid_xgboost.py`
- **Evaluation**: `h:\Axxend Project\BankFraudDetection\Anomally Detection\hybrid_evaluate.py`
- **Model**: `hybrid_xgboost_model.json` or `.joblib`

## 3. Data Pipeline & Feature Engineering
1.  **Reuse Preprocessing**: Use the same `load_and_preprocess_data` logic to ensure feature consistency.
2.  **Load UNSUPERVISED Models**: Load the trained `isolation_forest_model.joblib`, `autoencoder_model.joblib`, and `autoencoder_scaler.joblib`.
3.  **Generate Anomaly Features**:
    *   **IF Score**: `if_score = model_if.decision_function(X)` (Note: Invert if needed so higher = more anomalous).
    *   **AE Score**: `ae_score = MSE(X_scaled - model_ae.predict(X_scaled))` (Reconstruction Error).
    *   **Ensemble Score**: `avg_anomaly_score = (Normalize(if_score) + Normalize(ae_score)) / 2`.
4.  **Enrich Dataset**: Add `if_score`, `ae_score`, and `avg_anomaly_score` as NEW columns to the training data.

## 4. Model Architecture (XGBoost)
*   **Input**: Original Features + 3 New Anomaly Scores.
*   **Target**: `is_fraud` (Binary Classification).
*   **Algorithm**: XGBoost (Extreme Gradient Boosting).
*   **Objective**: `binary:logistic`.
*   **Evaluation Metric**: `aucpr` (Area Under Precision-Recall Curve) is best for imbalanced fraud data.

## 5. Implementation Steps
1.  **Setup**: Install `xgboost`.
2.  **Feature Generation**: Create a function `add_anomaly_scores(df, model_if, model_ae, scaler)` that returns the enriched dataframe.
3.  **Training**: Train XGBoost on the enriched dataset.
4.  **Evaluation**: Compare performance (Precision/Recall) of the Hybrid model vs. individual unsupervised models.
5.  **Visualization**: Feature Importance plot (to see if XGBoost actually relies on the anomaly scores).

Shall I proceed with **Step 1: Create the Hybrid Training Script**?
