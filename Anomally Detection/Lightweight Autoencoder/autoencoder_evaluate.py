import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_SET_AE_PATH = os.path.join(BASE_DIR, 'test_set_ae.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'autoencoder_model.joblib')
SCALER_PATH = os.path.join(BASE_DIR, 'autoencoder_scaler.joblib')
THRESHOLD_PATH = os.path.join(BASE_DIR, 'ae_threshold.txt')
CM_PLOT_PATH = os.path.join(BASE_DIR, 'ae_confusion_matrix.png')
ERROR_DIST_PATH = os.path.join(BASE_DIR, 'ae_reconstruction_error.png')

def evaluate_autoencoder():
    print("Loading Autoencoder Model, Scaler, and Test Data...")
    
    # Load Artifacts
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH) or not os.path.exists(THRESHOLD_PATH):
        raise FileNotFoundError("Missing model artifacts. Run train_light_autoencoder.py first.")
        
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    with open(THRESHOLD_PATH, 'r') as f:
        threshold = float(f.read().strip())
    print(f"Loaded Threshold: {threshold:.6f}")
    
    # Load Data
    df_test = pd.read_csv(TEST_SET_AE_PATH)
    X_test = df_test.drop('is_fraud', axis=1)
    y_test = df_test['is_fraud']
    
    print("Scaling Test Data...")
    X_test_scaled = scaler.transform(X_test)
    
    print("Calculating Reconstruction Error...")
    # Predict (Reconstruct)
    reconstructions = model.predict(X_test_scaled)
    # Calculate MSE per sample
    mse = np.mean(np.power(X_test_scaled - reconstructions, 2), axis=1)
    
    # Visualize Error Distribution (Normal vs Fraud)
    plt.figure(figsize=(10, 6))
    plot_df = pd.DataFrame({'Reconstruction Error': mse, 'Label': y_test})
    plot_df['Label'] = plot_df['Label'].map({0: 'Normal', 1: 'Fraud'})
    
    sns.histplot(data=plot_df, x='Reconstruction Error', hue='Label', common_norm=False, kde=True, bins=50)
    plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.2f})')
    plt.title('Autoencoder Reconstruction Error Distribution')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(ERROR_DIST_PATH)
    print(f"Error distribution plot saved to {ERROR_DIST_PATH}")
    
    # Classification based on Threshold
    # If Error > Threshold -> Fraud (1)
    y_pred = [1 if error > threshold else 0 for error in mse]
    
    # Metrics
    print("\n--- Model Predictions (Autoencoder) ---")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
    
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Predicted Normal', 'Predicted Fraud'], yticklabels=['Actual Normal', 'Actual Fraud'])
    plt.title('Autoencoder Confusion Matrix')
    plt.tight_layout()
    plt.savefig(CM_PLOT_PATH)
    print(f"Confusion matrix plot saved to {CM_PLOT_PATH}")

    print("\n--- Detailed Metrics ---")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    evaluate_autoencoder()
