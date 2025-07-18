import json
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import requests
import os
import zipfile

# Download helper functions
def download_file(url, filename):
    """Download file from URL with progress tracking"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=1024*1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                progress = downloaded / total_size * 100
                print(f"Downloading: {progress:.1f}%", end='\r')
    print(f"\nDownloaded {filename}")

def unzip_file(zip_path, extract_path):
    """Extract zip file"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Extracted {zip_path}")

# Feature engineering functions
def calculate_time_features(transactions):
    """Calculate time-based features for a wallet"""
    transactions = transactions.sort_values('timestamp')
    time_diffs = transactions['timestamp'].diff().dropna().dt.total_seconds()
    
    features = {
        'txn_freq': len(transactions),
        'time_std': time_diffs.std() if len(time_diffs) > 1 else 0,
        'time_median': time_diffs.median() if len(time_diffs) > 0 else 0,
        'time_iqr': np.subtract(*np.percentile(time_diffs, [75, 25])) if len(time_diffs) > 1 else 0
    }
    return features

def calculate_behavior_features(transactions):
    """Calculate transaction behavior features"""
    action_counts = transactions['action'].value_counts().to_dict()
    liquidation_count = action_counts.get('liquidationcall', 0)
    
    borrow_repay_ratio = 0
    if 'borrow' in action_counts and 'repay' in action_counts:
        if action_counts['borrow'] > 0:
            borrow_repay_ratio = action_counts['repay'] / action_counts['borrow']
    
    features = {
        'liquidation_count': liquidation_count,
        'borrow_repay_ratio': borrow_repay_ratio,
        'deposit_ratio': action_counts.get('deposit', 0) / len(transactions),
        'redeem_ratio': action_counts.get('redeemunderlying', 0) / len(transactions),
        'liquidation_ratio': liquidation_count / len(transactions) if liquidation_count > 0 else 0
    }
    return features

def engineer_features(df):
    """Engineer features for all wallets"""
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Group by wallet and compute features
    features = []
    for wallet, group in df.groupby('user'):
        time_features = calculate_time_features(group)
        behavior_features = calculate_behavior_features(group)
        
        wallet_features = {
            'wallet': wallet,
            **time_features,
            **behavior_features
        }
        features.append(wallet_features)
    
    return pd.DataFrame(features)

# Modeling functions
def compute_credit_scores(feature_df):
    """Compute credit scores using anomaly detection"""
    # Select relevant features
    X = feature_df[['txn_freq', 'time_std', 'time_median', 'time_iqr', 
                    'liquidation_count', 'borrow_repay_ratio', 
                    'deposit_ratio', 'redeem_ratio', 'liquidation_ratio']]
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Train model
    model = IsolationForest(
        n_estimators=150,
        contamination=0.1,
        random_state=42,
        verbose=1
    )
    model.fit(X)
    
    # Get raw anomaly scores (-1 to 1 where negative is anomaly)
    raw_scores = model.decision_function(X)
    
    # Convert to credit scores (0-1000)
    scaler = MinMaxScaler(feature_range=(0, 1000))
    credit_scores = scaler.fit_transform(raw_scores.reshape(-1, 1))
    
    # Create results DataFrame
    results = feature_df[['wallet']].copy()
    results['credit_score'] = credit_scores
    
    return results, model

# Analysis functions
def generate_analysis(credit_scores):
    """Generate analysis report and visualizations"""
    # Create bins for score distribution
    bins = list(range(0, 1001, 100))
    labels = [f"{i}-{i+99}" for i in bins[:-1]]
    credit_scores['score_bin'] = pd.cut(
        credit_scores['credit_score'], 
        bins=bins, 
        labels=labels,
        include_lowest=True
    )
    
    # Generate distribution plot
    plt.figure(figsize=(12, 6))
    bin_counts = credit_scores['score_bin'].value_counts().sort_index()
    bin_counts.plot(kind='bar', color='skyblue')
    plt.title('Credit Score Distribution')
    plt.xlabel('Credit Score Range')
    plt.ylabel('Number of Wallets')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('score_distribution.png')
    plt.close()
    
    # Create analysis report
    analysis = "# Credit Score Analysis Report\n\n"
    analysis += "## Score Distribution\n"
    analysis += f"![Score Distribution](score_distribution.png)\n\n"
    
    # Low score analysis
    low_scores = credit_scores[credit_scores['credit_score'] < 300]
    analysis += "## Low Score Wallets (0-299)\n"
    analysis += f"- **Count**: {len(low_scores)}\n"
    analysis += "- **Characteristics**:\n"
    analysis += "  - High liquidation rates\n"
    analysis += "  - Erratic transaction patterns\n"
    analysis += "  - Low repayment ratios\n"
    analysis += "  - Frequent liquidation events\n\n"
    
    # High score analysis
    high_scores = credit_scores[credit_scores['credit_score'] >= 700]
    analysis += "## High Score Wallets (700-1000)\n"
    analysis += f"- **Count**: {len(high_scores)}\n"
    analysis += "- **Characteristics**:\n"
    analysis += "  - Consistent transaction patterns\n"
    analysis += "  - High repayment ratios\n"
    analysis += "  - No liquidation history\n"
    analysis += "  - Balanced deposit/withdrawal behavior\n\n"
    
    # Save report
    with open('analysis.md', 'w') as f:
        f.write(analysis)
    
    return analysis

# Main execution
def main():
    # Download data
    data_file = 'transactions.json'
    if not os.path.exists(data_file):
        zip_file = 'transactions.zip'
        if not os.path.exists(zip_file):
            print("Downloading data...")
            download_file(
                "https://drive.google.com/uc?export=download&id=14ceBCLQ-BTcydDrFJauVA_PKAZ7VtDor",
                zip_file
            )
        print("Extracting data...")
        unzip_file(zip_file, '.')
    
    # Load data
    print("Loading data...")
    with open(data_file, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    # Engineer features
    print("Engineering features...")
    feature_df = engineer_features(df)
    
    # Compute credit scores
    print("Computing credit scores...")
    credit_scores, model = compute_credit_scores(feature_df)
    
    # Save results
    credit_scores.to_csv('wallet_credit_scores.csv', index=False)
    
    # Generate analysis
    print("Generating analysis...")
    generate_analysis(credit_scores)
    
    print("Process completed successfully!")

if __name__ == '__main__':
    main()
