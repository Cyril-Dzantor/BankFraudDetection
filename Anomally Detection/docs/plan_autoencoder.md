
# Implementation Plan: Lightweight Autoencoder for Anomaly Detection

## 1. Goal
To implement a **Lightweight Autoencoder** model (using `keras` or `sklearn`'s MLP/custom logic) that learns the representation of **Normal Transactions** and flags anomalies based on high **Reconstruction Error**.

## 2. Directory Structure
Work within the existing `Anomally Detection` folder.
- **Script**: `h:\Axxend Project\BankFraudDetection\Anomally Detection\train_light_autoencoder.py`
- **Evaluation**: `h:\Axxend Project\BankFraudDetection\Anomally Detection\autoencoder_evaluate.py`
- **Model**: `autoencoder_model.h5` or `.keras`

## 3. Data Pipeline
We will reuse the data preprocessing logic from the Isolation Forest workflow to ensure consistency (same features).
- **Features**: 
  - `tx_count_last_5m`, `tx_count_last_1h`, `tx_frequency_ratio`
  - `amount`, `avg_tx_amount_7d`, `amount_to_avg_ratio`
  - `failed_login_count_last_1h`
  - `accounts_per_device`, `accounts_per_ip_24h`
  - Encoded: `geo_country`, `currency`, `channel`, `auth_method`
  
- **Data Scaling**: 
  - **CRITICAL**: Autoencoders require normalized data (e.g., MinMax or StandardScaler) for the loss function (MSE) to work effectively. We will apply `StandardScaler`.
  
- **Training Set**: 
  - Train **ONLY on Normal Transactions** (`is_fraud == 0`). The autoencoder learns to reconstruct "normal" patterns.
  
- **Test Set**: 
  - Contains mixed Normal and Fraud transactions to evaluate detection capability.

## 4. Model Architecture (Lightweight)
A simple feed-forward neural network (MLP Autoencoder).
- **Input Layer**: `n_features`
- **Encoder**: 
  - Layer 1: Dense (e.g., 64 units, Relu)
  - Layer 2: Dense (e.g., 32 units, Relu) - *Bottleneck*
- **Decoder**:
  - Layer 3: Dense (e.g., 64 units, Relu)
  - Output Layer: Dense (`n_features`, Linear/Sigmoid depending on finding scaler)
  
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam

## 5. Anomaly Detection Logic
1.  Pass transaction $x$ through the trained autoencoder to get reconstruction $\hat{x}$.
2.  Calculate **Reconstruction Error** (MSE): $E = ||x - \hat{x}||^2$.
3.  Determine a **Threshold**:
    - Based on the reconstruction error distribution of the Training (Normal) set.
    - E.g., Threshold = Mean(Error) + 3 * Std(Error) or 95th percentile.
4.  **Prediction**:
    - If Error > Threshold -> **Anomaly (Fraud)**.
    - If Error <= Threshold -> **Normal**.

## 6. Implementation Steps
1.  **Setup**: Verify `tensorflow` / `keras` installation.
2.  **Preprocessing**: Copy/Import preprocessing from `train_isolation_forest.py`, add Scaling.
3.  **Training**: Build and train the model on Normal data.
4.  **Thresholding**: Calculate threshold on validation set.
5.  **Evaluation**: Evaluate on Test Set using Reconstruction Error.
6.  **Visualization**: Plot Error Distribution for Normal vs Fraud.

Shall I proceed with **Step 1: Setup and Preprocessing** (creating the script)?
