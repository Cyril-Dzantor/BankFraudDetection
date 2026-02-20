import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Define paths
# Using relative paths for portability
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', '..', 'production_fraud_dataset.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'isolation_forest_model.joblib')
CM_PLOT_PATH = os.path.join(BASE_DIR, 'confusion_matrix.png')

def load_and_preprocess_data(filepath):
    """
    Loads data, calculates contamination, and preprocesses based on specific user features.
    """
    print(f"Loading data from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
        
    df = pd.read_csv(filepath)
    
    # 1. Calculate Contamination Rate
    n_fraud = df['is_fraud'].sum()
    n_total = len(df)
    contamination = n_fraud / n_total
    print(f"Dataset Stats: {n_total} records, {n_fraud} frauds.")
    print(f"Calculated Contamination Rate: {contamination:.5f}")

    # 2. Select Specific Features
    # tx_count_last_5m, tx_count_last_1h, tx_frequency_ratio, amount,
    # avg_tx_amount_7d, amount_to_avg_ratio, failed_login_count_last_1h,
    # accounts_per_device, accounts_per_ip_24h
    # Categorical (to be encoded): geo_country, currency, channel, auth_method
    
    features_to_keep = [
        'tx_count_last_5m', 'tx_count_last_1h', 'tx_frequency_ratio', 
        'amount', 'avg_tx_amount_7d', 'amount_to_avg_ratio', 
        'failed_login_count_last_1h', 'accounts_per_device', 
        'accounts_per_ip_24h',
        'geo_country', 'currency', 'channel', 'auth_method'
    ]
    
    available_features = [f for f in features_to_keep if f in df.columns]
    print(f"Selected Features for Training: {available_features}")
    
    # Create clean dataframe with selected features + target
    df_clean = df[available_features + ['is_fraud']].copy()
    
    # 3. Encode Categorical Columns
    # User specified these should be encoded
    cat_cols = ['geo_country', 'currency', 'channel', 'auth_method']
    
    for col in cat_cols:
        if col in df_clean.columns:
            # Fill NaNs with 'Unknown' ensuring string type
            df_clean[col] = df_clean[col].fillna('Unknown').astype(str)
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col])

    # 4. Handle Missing Values
    df_clean = df_clean.fillna(0)
    
    # 5. Separate Target
    X = df_clean.drop('is_fraud', axis=1)
    y = df_clean['is_fraud']
    
    return X, y, contamination

def train_model(X, y, contamination):
    """
    Trains Isolation Forest and evaluates against ground truth.
    """
    # Split Data: Train on one part, Test on another
    # Note: Unsupervised training typically uses only 'normal' data or full dirty data.
    # Here we train on X_train (unsupervised mode) which contains both normal and anomalies.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save Test Set for separate evaluation
    print("Saving Test Set to test_set.csv...")
    test_df = X_test.copy()
    test_df['is_fraud'] = y_test
    test_df.to_csv(os.path.join(BASE_DIR, '..', 'data', 'test_set.csv'), index=False)
    
    print("Training Isolation Forest...")
    # Initialize model
    # n_jobs=-1 uses all CPU cores
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    # Train
    iso_forest.fit(X_train)
    
    return iso_forest

def main():
    print("Starting Fraud Detection Model Training...")
    
    # 1. Load & Preprocess
    try:
        X, y, contamination = load_and_preprocess_data(DATA_PATH)
    except Exception as e:
        print(f"Error during data loading/preprocessing: {e}")
        return

    # 2. Train & Evaluate
    model = train_model(X, y, contamination)
    
    # 3. Save Model
    print(f"\nSaving model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    print("Done.")

if __name__ == "__main__":
    main()
