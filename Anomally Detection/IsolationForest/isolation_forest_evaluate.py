import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'isolation_forest_model.joblib')
TEST_DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'test_set.csv')
SCORE_PLOT_PATH = os.path.join(BASE_DIR, 'anomaly_score_distribution.png')
CM_PLOT_PATH = os.path.join(BASE_DIR, 'confusion_matrix.png')

def evaluate_model():
    print("Loading model and test data...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"Test data not found at {TEST_DATA_PATH}")

    # Load Model
    model = joblib.load(MODEL_PATH)
    
    # Load Test Data
    df_test = pd.read_csv(TEST_DATA_PATH)
    X_test = df_test.drop('is_fraud', axis=1)
    y_test = df_test['is_fraud']

    print("Calculating Anomaly Scores...")
    # decision_function outputs anomaly score
    # Positive scores: Normal
    # Negative scores: Anomaly
    scores = model.decision_function(X_test)
    
    # Display Score Statistics
    print("\n--- Anomaly Score Statistics ---")
    print(pd.Series(scores).describe())
    
    # Save/Display Score Distribution
    plt.figure(figsize=(10, 6))
    plot_df = pd.DataFrame({'Score': scores, 'Label': y_test})
    plot_df['Label'] = plot_df['Label'].map({0: 'Normal', 1: 'Fraud'})
    
    sns.histplot(data=plot_df, x='Score', hue='Label', common_norm=False, kde=True, bins=50)
    plt.title('Isolation Forest Anomaly Score Distribution')
    plt.xlabel('Anomaly Score (Negative = More Anomalous)')
    plt.tight_layout()
    plt.savefig(SCORE_PLOT_PATH)
    print(f"Score distribution plot saved to {SCORE_PLOT_PATH}")

    # Standard Evaluation (Predictions)
    print("\n--- Model Predictions ---")
    y_pred_iso = model.predict(X_test)
    y_pred = [1 if p == -1 else 0 for p in y_pred_iso]

    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
    
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Predicted Normal', 'Predicted Fraud'], yticklabels=['Actual Normal', 'Actual Fraud'])
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(CM_PLOT_PATH)

    print("\n--- Detailed Metrics ---")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    evaluate_model()
