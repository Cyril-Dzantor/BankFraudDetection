# Data Integration Specification: Banking to Cognize

To enable real-time fraud scoring and network analysis, the core banking system (or a middleware layer) must provide a JSON payload for every transaction. This data is divided into three categories:

## 1. Core Transaction Data (Raw)
These are standard fields available at the moment of the transaction.

| Field | Type | Description |
| :--- | :--- | :--- |
| `amount` | Float | The transaction value in the local currency. |
| `currency` | String | ISO currency code (e.g., "GHS", "USD"). |
| `channel` | Enum | One of: `mobile`, `web`, `atm`, `pos`. |
| `auth_method`| Enum | One of: `pin`, `biometric`, `otp`, `password`. |
| `geo_country` | String | Two-letter country code (ISO 3166-1 alpha-2). |
| `acctStart` | String | Source account identifier (e.g., account number). |
| `device_id` | String | A unique hardware identifier for the mobile/web device. |

## 2. Behavioral Features (Aggregated)
Our AI model requires these calculated signals to identify anomalies. These are usually computed by the bank's data layer over a rolling window.

| Field | Type | Description |
| :--- | :--- | :--- |
| `tx_count_last_5m` | Float | Number of transactions from this account in the last 5 minutes. |
| `tx_count_last_1h` | Float | Number of transactions from this account in the last 1 hour. |
| `avg_tx_amount_7d` | Float | The average transaction amount for this user over the last 7 days. |
| `amount_to_avg_ratio`| Float | Current `amount` divided by `avg_tx_amount_7d`. |
| `tx_frequency_ratio` | Float | Current frequency vs historically expected frequency. |

## 3. Security & Contextual Signals
Used for both AI scoring and Link Analysis (Graph).

| Field | Type | Description |
| :--- | :--- | :--- |
| `new_device_flag` | Int | `1` if the device has never been seen on this account before. |
| `country_mismatch` | Int | `1` if the transaction location is unusual for this user. |
| `failed_login_1h` | Float | Number of failed login/auth attempts in the last hour. |
| `accounts_per_ip` | Float | Number of unique accounts seen from this IP in the last 24h. |

---

## 🔗 Endpoint Implementation
The bank should `POST` this payload to:
`http://<your-backend>/api/v1/alerts/score`

### Example Payload
```json
{
  "acctStart": "ACCT-88219",
  "amount": 2500.00,
  "currency": "GHS",
  "channel": "mobile",
  "auth_method": "biometric",
  "geo_country": "GH",
  "tx_count_last_5m": 1,
  "avg_tx_amount_7d": 1200.0,
  "amount_to_avg_ratio": 2.08,
  "new_device_flag": 1
}
```
