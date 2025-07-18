# Wallet Credit Scoring System

## Overview
This solution calculates credit scores (0-1000) for cryptocurrency wallets based on their transaction behavior in the Aave V2 protocol. Higher scores indicate reliable behavior, while lower scores indicate risky or bot-like activity.

## Methodology

### Feature Engineering
- **Time Features**:
  - Transaction frequency
  - Time between transactions (std, median, IQR)
- **Behavior Features**:
  - Liquidation count and ratio
  - Borrow/repay ratio
  - Deposit/redeem ratios

### Scoring Model
- Uses Isolation Forest algorithm for anomaly detection
- Normalizes scores to 0-1000 range
- Key risk indicators:
  - High liquidation rates
  - Erratic transaction timing
  - Low repayment ratios
  - Frequent liquidation events

### Model Transparency
- Scores are relative to overall wallet behavior
- Higher scores indicate:
  - Consistent transaction patterns
  - Complete repayments
  - No liquidation history
- Lower scores indicate:
  - Irregular transaction timing
  - High liquidation rates
  - Low repayment ratios

## Processing Flow
1. Download and prepare transaction data
2. Engineer time and behavior features
3. Train Isolation Forest model
4. Calculate anomaly scores
5. Normalize scores to 0-1000 range
6. Generate score distribution analysis

## Usage
1. Install requirements: `pip install pandas numpy scikit-learn matplotlib requests`
2. Run script: `python generate_scores.py`
3. Outputs:
   - `wallet_credit_scores.csv`: Wallet addresses with scores
   - `analysis.md`: Score distribution analysis
   - `score_distribution.png`: Visualization of score distribution

## Extensibility
- Add new features:
  - Transaction amount statistics
  - Protocol interaction patterns
  - Time-of-day patterns
- Incorporate additional data sources:
  - Other DeFi protocols
  - On-chain reputation systems
- Implement model versioning
- Add temporal analysis for score trends
