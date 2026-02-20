import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', '..', 'production_fraud_dataset.csv')

# Output Paths
MODEL_PATH = os.path.join(BASE_DIR, 'xgboost_model.joblib')
FEATURE_IMPORTANCE_PATH = os.path.join(BASE_DIR, 'xgboost_feature_importance.png')
CM_PLOT_PATH = os.path.join(BASE_DIR, 'xgboost_confusion_matrix.png')
TEST_SET_XGB_PATH = os.path.join(BASE_DIR, '..', 'data', 'test_set_xgb.csv')

# Unsupervised Model Paths (Inputs)
IF_MODEL_PATH = os.path.join(BASE_DIR, '..', 'IsolationForest', 'isolation_forest_model.joblib')
AE_MODEL_PATH = os.path.join(BASE_DIR, '..', 'Lightweight Autoencoder', 'autoencoder_model.joblib')
AE_SCALER_PATH = os.path.join(BASE_DIR, '..', 'Lightweight Autoencoder', 'autoencoder_scaler.joblib')

def load_and_preprocess_data(filepath):
    """
    Loads data, generates anomaly scores from IF/AE, and preprocesses for XGBoost.
    """
    print(f"Loading data from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
        
    df = pd.read_csv(filepath)
    print(f"Dataset Shape: {df.shape}")

    # 1. Define Feature Sets
    # Features used by Isolation Forest and Autoencoder (Must match their training!)
    unsupervised_features = [
        'tx_count_last_5m', 'tx_count_last_1h', 'tx_frequency_ratio', 
        'amount', 'avg_tx_amount_7d', 'amount_to_avg_ratio', 
        'failed_login_count_last_1h', 'accounts_per_device', 
        'accounts_per_ip_24h',
        'geo_country', 'currency', 'channel', 'auth_method'
    ]
    
    # Full Feature Set for XGBoost (User Requested)
    xgboost_features = [
        'tx_count_last_5m', 'tx_count_last_1h', 'tx_frequency_ratio',
        'amount', 'avg_tx_amount_7d', 'amount_to_avg_ratio',
        'new_device_flag', 'device_seen_before', 'country_mismatch_flag',
        'accounts_per_device', 'accounts_per_ip_24h',
        'failed_login_count_last_1h', 'failed_logins_then_success',
        'channel', 'auth_method', 'geo_country', 'currency'
    ]
    
    # 2. Preprocess for Unsupervised Models (Encoding)
    # We need a temporary dataframe encoded exactly as IF/AE expect
    df_unsup = df[unsupervised_features].copy()
    cat_cols = ['geo_country', 'currency', 'channel', 'auth_method']
    for col in cat_cols:
        if col in df_unsup.columns:
            df_unsup[col] = df_unsup[col].fillna('Unknown').astype(str)
            le = LabelEncoder()
            df_unsup[col] = le.fit_transform(df_unsup[col])
    df_unsup = df_unsup.fillna(0)

    # 3. Generate Anomaly Scores
    print("Generating Anomaly Scores from Unsupervised Models...")
    
    # Isolation Forest Score
    if os.path.exists(IF_MODEL_PATH):
        print("Loading Isolation Forest...")
        if_model = joblib.load(IF_MODEL_PATH)
        # decision_function: lower = more anomalous (negative)
        # We might want to invert it so higher = anomaly, but XGBoost can handle it.
        # Let's keep it raw.
        df['isolation_forest_score'] = if_model.decision_function(df_unsup)
    else:
        print("Warning: Isolation Forest model not found. Filling with 0.")
        df['isolation_forest_score'] = 0

    # Autoencoder Reconstruction Error
    if os.path.exists(AE_MODEL_PATH) and os.path.exists(AE_SCALER_PATH):
        print("Loading Autoencoder...")
        ae_model = joblib.load(AE_MODEL_PATH)
        ae_scaler = joblib.load(AE_SCALER_PATH)
        
        # Scale data
        X_scaled = ae_scaler.transform(df_unsup)
        # Reconstruct
        reconstructions = ae_model.predict(X_scaled)
        # MSE Error
        mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
        df['autoencoder_reconstruction_error'] = mse
    else:
        print("Warning: Autoencoder model/scaler not found. Filling with 0.")
        df['autoencoder_reconstruction_error'] = 0

    # 4. Prepare Final XGBoost Dataset
    # Add anomaly scores to feature list
    final_features = xgboost_features + ['isolation_forest_score', 'autoencoder_reconstruction_error']
    
    print(f"Final Feature List for XGBoost ({len(final_features)} features):")
    print(final_features)
    
    df_final = df[final_features + ['is_fraud']].copy()
    
    # Encode Categoricals for XGBoost
    # Note: We re-encode here because the main df wasn't encoded yet
    for col in cat_cols:
        if col in df_final.columns:
            df_final[col] = df_final[col].fillna('Unknown').astype(str)
            le = LabelEncoder()
            df_final[col] = le.fit_transform(df_final[col])
            
    # Handle Missing
    df_final = df_final.fillna(0)
    
    return df_final

def train_xgboost(df):
    """
    Trains an XGBoost Classifier on the enriched dataset.
    """
    # Split into features and target
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    
    # Calculate scale_pos_weight
    ratio = float(np.sum(y == 0)) / np.sum(y == 1)
    print(f"Class Imbalance Ratio (Neg/Pos): {ratio:.2f}")
    
    # Split into Train and Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training on {len(X_train)} transactions.")
    
    # Define XGBoost Classifier
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=ratio, 
        eval_metric='aucpr',
        random_state=42,
        n_jobs=-1
    )
    
    print("Starting Training (XGBoost Hybrid)...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Get Feature Importance
    importance = model.feature_importances_
    features = X.columns
    indices = np.argsort(importance)[::-1]
    
    # Plot Feature Importance
    plt.figure(figsize=(12, 10))
    plt.title("Hybrid XGBoost Feature Importance")
    plt.bar(range(X.shape[1]), importance[indices], align="center")
    plt.xticks(range(X.shape[1]), features[indices], rotation=90, ha='right')
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PATH)
    print(f"Feature importance plot saved to {FEATURE_IMPORTANCE_PATH}")
    
    # Save Test Set for Evaluation
    print("Saving Test Set for Evaluation...")
    test_df_save = X_test.copy()
    test_df_save['is_fraud'] = y_test
    test_df_save.to_csv(TEST_SET_XGB_PATH, index=False)

    return model

def evaluate_xgboost(model, df):
    # Quick evaluation
    test_df = pd.read_csv(TEST_SET_XGB_PATH)
    X_test = test_df.drop('is_fraud', axis=1)
    y_test = test_df['is_fraud']
    
    print("Evaluating on Test Set...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
    
    cm = confusion_matrix(y_test, y_pred)
    # print("Confusion Matrix:\n", cm)
    
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Normal', 'Predicted Fraud'], yticklabels=['Actual Normal', 'Actual Fraud'])
    plt.title('Hybrid XGBoost Confusion Matrix')
    plt.tight_layout()
    plt.savefig(CM_PLOT_PATH)

def main():
    print("Initializing Hybrid XGBoost Training...")
    
    try:
        df = load_and_preprocess_data(DATA_PATH)
    except Exception as e:
        print(f"Error loading data: {e}")
        # import traceback
        # traceback.print_exc()
        return

    model = train_xgboost(df)
    
    evaluate_xgboost(model, df)
    
    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    print("Done. Model saved.")

if __name__ == "__main__":
    main()
