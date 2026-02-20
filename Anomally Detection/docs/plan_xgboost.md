
# Implementation Plan: Standalone XGBoost Classifier

## 1. Goal
To train a **Supervised XGBoost Classifier** on the fraud dataset.
This will serve as the **final decision maker** in the pipeline. It learns explicit patterns (e.g., specific rules, thresholds) from fraud vs. normal examples.

## 2. Directory Structure
Work within the existing `Anomally Detection` folder.
- **Script**: `h:\Axxend Project\BankFraudDetection\Anomally Detection\train_xgboost.py`
- **Evaluation**: `h:\Axxend Project\BankFraudDetection\Anomally Detection\xgboost_evaluate.py`
- **Model**: `xgboost_model.json` (XGBoost JSON format)

## 3. Data Pipeline
1.  **Reuse Preprocessing**: Consistently use `load_and_preprocess_data` logic to ensure fairness.
2.  **Imbalance Handling**: Fraud data is highly imbalanced (~4% fraud). We might need to adjust `scale_pos_weight` in XGBoost (ratio of negative to positive samples) or use metrics like `aucpr`.
3.  **Data Split**: Standard 80/20 train/test.

## 4. Model Architecture (XGBoost)
*   **Estimator**: `XGBClassifier`
*   **Hyperparameters**:
    *   `n_estimators`: 100-200 (Number of trees)
    *   `max_depth`: 3-6 (Tree complexity)
    *   `learning_rate`: 0.1 (Step size)
    *   `subsample`: 0.8 (Prevent overfitting)
    *   `scale_pos_weight`: ~25 (To handle imbalance, calculated as `neg/pos`)
*   **Objective**: `binary:logistic` (Output probability)
*   **Eval Metric**: `auc`, `aucpr`

## 5. Implementation Steps
1.  **Setup**: Install `xgboost`.
2.  **Training**: Train the model on the preprocessed dataset.
3.  **Evaluation**: Compare performance (Precision/Recall).
4.  **Feature Importance**: XGBoost provides built-in feature importance. This is crucial for explainability.
5.  **Artifacts**: Save the model and feature importance plot.

Shall I proceed with **Step 1: Setup and Training Script**?
