import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import requests
import json
import time
from datetime import datetime, timedelta
import concurrent.futures
import joblib

# Load the model for local prediction (control group)
model = joblib.load("./models/Optimized XGBoost.pkl")
preprocessing_pipeline = joblib.load("./models/feature_engineering_pipeline.pkl")

# Configuration
API_URL = "http://localhost:8000"  # URL for the deployed model (treatment group)
TRANSACTION_DATA_PATH = "./data/test_transactions.csv"
OUTPUT_DIR = "./data/ab_test_results"
TEST_DURATION_DAYS = 7
TRANSACTIONS_PER_DAY = 1000

# Ensure output directory exists
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_test_transactions(num_transactions=1000):
        """
        Args:
            num_transactions (int): Number of transactions to generate
            
        Returns:
            pd.DataFrame: DataFrame with synthetic transaction data
        """
        # Define possible values for categorical features
        payment_methods = ["credit_card", "debit_card", "bank_transfer", "mobile_payment"]
        device_types = ["mobile", "web", "pos", "tablet"]
        customer_locations = ["urban", "suburban", "rural"]
        transaction_results = ["success", "pending", "failed"]
        latency_bins = ["low", "medium", "high"]
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        amount_bins = ["low", "medium", "high"]
        
        # Generate random data
        num_transactions = num_transactions
        data = {
            "transaction_id": [f"tx-{i}" for i in range(1, num_transactions + 1)],
            "timestamp": np.random.choice(pd.date_range("2025-03-01", periods=num_transactions, freq="H").astype(str)),
            "merchant_id": [f"MERCH{np.random.randint(10000, 99999)}" for _ in range(num_transactions)],
            "customer_id": [f"CUST{np.random.randint(100000, 999999)}" for _ in range(num_transactions)],
            "customer_location": np.random.choice(customer_locations, size=num_transactions),
            "payment_amount": np.random.gamma(shape=3, scale=100, size=num_transactions),
            "payment_method": np.random.choice(payment_methods, size=num_transactions),
            "device_type": np.random.choice(device_types, size=num_transactions),
            "network_latency": np.random.uniform(50, 150, size=num_transactions),
            "result": np.random.choice(transaction_results, size=num_transactions),
            "latency_bin_encoded": np.random.randint(1, 4, size=num_transactions),
            "network_latency_scaled": np.random.uniform(0, 1, size=num_transactions),
            "merchant_rolling_avg_amount_scaled": np.random.uniform(0, 1, size=num_transactions),
            "merchant_success_rate_scaled": np.random.uniform(0, 1, size=num_transactions),
            "device_success_rate_scaled": np.random.uniform(0, 1, size=num_transactions),
            "payment_method_rolling_success_scaled": np.random.uniform(0, 1, size=num_transactions),
            "location_success_rate_scaled": np.random.uniform(0, 1, size=num_transactions),
            "payment_location_success_rate_scaled": np.random.uniform(0, 1, size=num_transactions),
            "merchant_transaction_count_log": np.random.uniform(2, 6, size=num_transactions),
            "hourly_transaction_volume_log": np.random.uniform(4, 8, size=num_transactions),
            "amount_log": np.log1p(np.random.gamma(shape=3, scale=100, size=num_transactions)),
            "merchant_rolling_avg_amount": np.random.uniform(50, 500, size=num_transactions),
            "hourly_transaction_volume": np.random.randint(500, 2000, size=num_transactions),
            "merchant_success_rate": np.random.uniform(0.85, 0.99, size=num_transactions),
            "time_of_day": np.random.choice(["morning", "afternoon", "evening", "night"], size=num_transactions),
            "latency_bin": np.random.choice(latency_bins, size=num_transactions),
            "day_name": np.random.choice(day_names, size=num_transactions),
            "amount_bin": np.random.choice(amount_bins, size=num_transactions),
        }

        return pd.DataFrame(data)

def old_routing_strategy(transaction):
    """
    Simulate the old routing strategy
    
    Args:
        transaction (dict): Transaction data
        
    Returns:
        dict: Routing decision
    """
    # Simple rule-based routing
    if transaction["payment_amount"] > 1000:
        return {
            "transaction_id": transaction["transaction_id"],
            "success_probability": 0.5,
            "recommended_action": "review",
            "strategy": "old"
        }
    elif transaction["previous_failures"] > 0:
        return {
            "transaction_id": transaction["transaction_id"],
            "success_probability": 0.4,
            "recommended_action": "review",
            "strategy": "old"
        }
    else:
        return {
            "transaction_id": transaction["transaction_id"],
            "success_probability": 0.7,
            "recommended_action": "route",
            "strategy": "old"
        }

def new_routing_strategy(transaction):
    """
    Use the deployed model API for routing decision
    
    Args:
        transaction (dict): Transaction data
        
    Returns:
        dict: Routing decision
    """
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=transaction,
            timeout=2
        )
        
        if response.status_code == 200:
            result = response.json()
            result["strategy"] = "new"
            return result
        else:
            # Fallback to old strategy if API fails
            return old_routing_strategy(transaction)
    except:
        # Fallback to old strategy if API fails
        return old_routing_strategy(transaction)

def simulate_transaction_outcome(transaction, routing_decision):
    """
    Simulate the actual outcome of a transaction based on the routing decision
    
    Args:
        transaction (dict): Transaction data
        routing_decision (dict): Routing decision
        
    Returns:
        dict: Transaction outcome
    """
    # Use the predicted success probability as the baseline
    base_success_prob = routing_decision["success_probability"]
    
    # Adjust based on the recommended action
    if routing_decision["recommended_action"] == "review" and base_success_prob < 0.6:
        # If we correctly identified a risky transaction and reviewed it
        adjusted_prob = min(0.9, base_success_prob + 0.3)
    elif routing_decision["recommended_action"] == "route" and base_success_prob > 0.7:
        # If we correctly identified a safe transaction and routed it directly
        adjusted_prob = base_success_prob
    else:
        # If we made a sub-optimal decision
        adjusted_prob = max(0.1, base_success_prob - 0.1)
    
    # Simulate outcome based on adjusted probability
    success = np.random.random() < adjusted_prob
    
    return {
        "transaction_id": transaction["transaction_id"],
        "strategy": routing_decision["strategy"],
        "recommended_action": routing_decision["recommended_action"],
        "predicted_probability": routing_decision["success_probability"],
        "success": success,
        "processing_time": np.random.uniform(0.5, 3.0)  # Simulate processing time in seconds
    }

def analyze_results(results_df):
    """
    Analyze A/B test results
    
    Args:
        results_df (pd.DataFrame): DataFrame with test results
    """
    print("\nAnalyzing A/B test results...")
    
    # Group by strategy
    grouped = results_df.groupby("strategy")
    
    # Calculate key metrics
    success_rates = grouped["success"].mean()
    processing_times = grouped["processing_time"].mean()
    
    # Print summary
    print("\nSummary of A/B Test Results:")
    print(f"Number of transactions: {len(results_df)}")
    print(f"Old strategy transactions: {sum(results_df['strategy'] == 'old')}")
    print(f"New strategy transactions: {sum(results_df['strategy'] == 'new')}")
    print("\nSuccess Rates:")
    for strategy, rate in success_rates.items():
        print(f"  {strategy.capitalize()} strategy: {rate:.4f} ({rate*100:.2f}%)")
    
    print("\nAverage Processing Times:")
    for strategy, time in processing_times.items():
        print(f"  {strategy.capitalize()} strategy: {time:.2f} seconds")
    
    # Statistical testing
    old_results = results_df[results_df["strategy"] == "old"]["success"]
    new_results = results_df[results_df["strategy"] == "new"]["success"]
    
    # Chi-square test for success rates
    contingency = pd.crosstab(results_df["strategy"], results_df["success"])
    chi2, p_value, _, _ = stats.chi2_contingency(contingency)
    
    print("\nStatistical Tests:")
    print(f"Chi-square test for success rates: chi2={chi2:.4f}, p={p_value:.4f}")
    print(f"Statistically significant difference: {'Yes' if p_value < 0.05 else 'No'}")
    
    # T-test for processing times
    old_times = results_df[results_df["strategy"] == "old"]["processing_time"]
    new_times = results_df[results_df["strategy"] == "new"]["processing_time"]
    t_stat, p_value = stats.ttest_ind(old_times, new_times)

    print(f"T-test for processing times: t={t_stat:.4f}, p={p_value:.4f}")
    print(f"Statistically significant difference: {'Yes' if p_value < 0.05 else 'No'}")

    # Calculate business impact
    old_success_rate = old_results.mean()
    new_success_rate = new_results.mean()
    improvement = new_success_rate - old_success_rate
    relative_improvement = improvement / old_success_rate

    print("\nBusiness Impact:")
    print(f"Absolute improvement in success rate: {improvement:.4f} ({improvement*100:.2f}%)")
    print(f"Relative improvement: {relative_improvement:.4f} ({relative_improvement*100:.2f}%)")

def create_visualizations(results_df):
    """
    Create visualizations based on A/B test results
    
    Args:
        results_df (pd.DataFrame): DataFrame with test results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set style
    sns.set(style="whitegrid")

    # Success rate comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x="strategy", y="success", data=results_df, ci=95)
    plt.title("Transaction Success Rate by Routing Strategy", fontsize=16)
    plt.xlabel("Strategy", fontsize=14)
    plt.ylabel("Success Rate", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(f"{OUTPUT_DIR}/success_rate_comparison_{timestamp}.png", dpi=300, bbox_inches="tight")

    # Processing time comparison
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="strategy", y="processing_time", data=results_df)
    plt.title("Transaction Processing Time by Routing Strategy", fontsize=16)
    plt.xlabel("Strategy", fontsize=14)
    plt.ylabel("Processing Time (seconds)", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(f"{OUTPUT_DIR}/processing_time_comparison_{timestamp}.png", dpi=300, bbox_inches="tight")

    # Success rate by predicted probability
    plt.figure(figsize=(10, 6))
    sns.lineplot(x="predicted_probability", y="success", data=results_df, hue="strategy")
    plt.title("Actual Success Rate by Predicted Probability", fontsize=16)
    plt.xlabel("Predicted Success Probability", fontsize=14)
    plt.ylabel("Actual Success Rate", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(f"{OUTPUT_DIR}/calibration_curve_{timestamp}.png", dpi=300, bbox_inches="tight")

    print(f"\nVisualizations saved to {OUTPUT_DIR}/ directory")
    plt.show()
        
def run_ab_test():
    """
    Run A/B test comparing old vs. new routing strategies
    """
    print("Starting A/B test simulation...")
    
    # Load test transaction data
    try:
        test_transactions = pd.read_csv(TRANSACTION_DATA_PATH)
    except:
        # Generate synthetic test data if file doesn't exist
        print("Test data file not found... generating synthetic data...")
        test_transactions = generate_test_transactions()
    
    # Initialize results storage
    results = []
    
    # Process transactions
    print(f"Processing {len(test_transactions)} transactions...")
    
    for i, row in test_transactions.iterrows():
        transaction = row.to_dict()
        
        # A/B test assignment (50/50 split)
        if np.random.random() < 0.5:
            # Control group - old strategy
            routing_decision = old_routing_strategy(transaction)
        else:
            # Treatment group - new strategy
            routing_decision = new_routing_strategy(transaction)
        
        # Simulate outcome
        outcome = simulate_transaction_outcome(transaction, routing_decision)
        results.append(outcome)
        
        # Log progress
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} transactions...")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save raw results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f"{OUTPUT_DIR}/ab_test_results_{timestamp}.csv", index=False)
    
    # Analyze results
    analyze_results(results_df)


    

if __name__ == "__main__":
    run_ab_test()   