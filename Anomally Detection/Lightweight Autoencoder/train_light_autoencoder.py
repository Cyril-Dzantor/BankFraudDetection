import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor
import joblib
import matplotlib.pyplot as plt

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'production_fraud_dataset.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'autoencoder_model.joblib') # Saving sklearn model as joblib
SCALER_PATH = os.path.join(BASE_DIR, 'autoencoder_scaler.joblib')
LOSS_PLOT_PATH = os.path.join(BASE_DIR, 'autoencoder_loss.png')
TEST_SET_AE_PATH = os.path.join(BASE_DIR, 'test_set_ae.csv')
THRESHOLD_PATH = os.path.join(BASE_DIR, 'ae_threshold.txt')

def load_and_preprocess_data(filepath):
    """
    Loads data and preprocesses specifically for Autoencoder training.
    """
    print(f"Loading data from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
        
    df = pd.read_csv(filepath)
    print(f"Dataset Shape: {df.shape}")

    # 1. Feature Selection (Same as Isolation Forest)
    features_to_keep = [
        'tx_count_last_5m', 'tx_count_last_1h', 'tx_frequency_ratio', 
        'amount', 'avg_tx_amount_7d', 'amount_to_avg_ratio', 
        'failed_login_count_last_1h', 'accounts_per_device', 
        'accounts_per_ip_24h',
        'geo_country', 'currency', 'channel', 'auth_method'
    ]
    
    available_features = [f for f in features_to_keep if f in df.columns]
    print(f"Selected Features: {available_features}")
    
    df_clean = df[available_features + ['is_fraud']].copy()
    
    # 2. Encode Categorical Columns
    cat_cols = ['geo_country', 'currency', 'channel', 'auth_method']
    for col in cat_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('Unknown').astype(str)
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col])

    # 3. Handle Missing Values
    df_clean = df_clean.fillna(0)
    
    return df_clean

def train_autoencoder_sklearn(df):
    """
    Trains an Autoencoder using Scikit-Learn's MLPRegressor.
    Architecture: [16, 8, 16] hidden layers.
    """
    # Split into features and target
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    
    # Split into Train and Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Filter Training Data: Train ONLY on Normal transactions
    train_normal_indices = y_train[y_train == 0].index
    X_train_normal = X_train.loc[train_normal_indices]
    
    print(f"Training on {len(X_train_normal)} normal transactions.")
    
    # Scaling
    print("Scaling data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_normal) # Fit on Normal
    X_test_scaled = scaler.transform(X_test) # Transform Test
    
    # Define MLP Autoencoder Architecture
    # Input -> [16, 8, 16] -> Output
    # Hidden layers are (16, 8, 16).
    # MLPRegressor learns f(X) = y. Ideally f(X) = X.
    
    autoencoder = MLPRegressor(
        hidden_layer_sizes=(16, 8, 16),
        activation='relu',
        solver='adam',
        batch_size=64,
        max_iter=50, # Epochs equivalent
        random_state=42,
        verbose=True,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    print("Starting Training (MLPRegressor)...")
    # Train to map input to input
    autoencoder.fit(X_train_scaled, X_train_scaled)
    
    # Plot Training Curve
    # MLPRegressor stores loss curve in loss_curve_
    plt.figure(figsize=(10, 6))
    plt.plot(autoencoder.loss_curve_, label='Training Loss')
    # Validation scores are stored in validation_scores_ if early_stopping=True
    if hasattr(autoencoder, 'validation_scores_') and autoencoder.validation_scores_:
        # Note: Validation scores are R^2 score for regressor, higher is better.
        # Loss is decreased, Score is increased. To plot on same graph is tricky.
        # Just plot Training Loss for now.
        pass

    plt.title('Autoencoder Training Loss (MLP)')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(LOSS_PLOT_PATH)
    print(f"Loss plot saved to {LOSS_PLOT_PATH}")
    
    # Save Test Set
    print("Saving Test Set for Evaluation...")
    test_df_save = X_test.copy()
    test_df_save['is_fraud'] = y_test
    test_df_save.to_csv(TEST_SET_AE_PATH, index=False)
    
    # Determine Threshold
    print("Calculating Threshold...")
    reconstructions = autoencoder.predict(X_train_scaled)
    train_loss = np.mean(np.power(X_train_scaled - reconstructions, 2), axis=1)
    threshold = np.mean(train_loss) + 3 * np.std(train_loss)
    print(f"Reconstruction Error Threshold: {threshold:.6f}")
    
    # Save Threshold
    with open(THRESHOLD_PATH, 'w') as f:
        f.write(str(threshold))
        
    return autoencoder, scaler

def main():
    print("Initializing Lightweight Autoencoder (Sklearn MLP)...")
    
    try:
        df = load_and_preprocess_data(DATA_PATH)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    model, scaler = train_autoencoder_sklearn(df)
    
    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    
    print(f"Saving scaler to {SCALER_PATH}...")
    joblib.dump(scaler, SCALER_PATH)
    
    print("Done. Model, scaler, and threshold saved.")

if __name__ == "__main__":
    main()
