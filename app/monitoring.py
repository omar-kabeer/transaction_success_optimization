import pandas as pd
import numpy as np
import logging
import json
import time
import requests
import os
from datetime import datetime, timedelta
import boto3
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("monitoring.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
API_URL = "http://localhost:8000"  # URL for the deployed model API
MONITORING_OUTPUT_DIR = "./monitoring_results"
CLOUDWATCH_LOG_GROUP = "/transaction-predictor/api-logs"
FEEDBACK_DIR = "./feedback"
MODEL_RETRAINING_THRESHOLD = 0.05  # Retrain if drift exceeds 5%

# Ensure directories exist
os.makedirs(MONITORING_OUTPUT_DIR, exist_ok=True)
os.makedirs(FEEDBACK_DIR, exist_ok=True)

class ModelMonitor:
    def __init__(self):
        self.api_url = API_URL
        self.output_dir = MONITORING_OUTPUT_DIR
        self.cloudwatch_client = boto3.client('logs')
    
    def check_api_health(self):
        """Check if the API is healthy"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"API health check successful: {health_data}")
                return True, health_data
            else:
                logger.error(f"API health check failed: Status code {response.status_code}")
                return False, None
        except Exception as e:
            logger.error(f"API health check failed: {str(e)}")
            return False, None
    
    def monitor_response_times(self, n_samples=100):
        """Monitor API response times"""
        logger.info(f"Monitoring API response times with {n_samples} sample requests")
        
        # Generate sample transactions
        sample_transactions = generate_random_transactions(n_samples)
        
        # Test API response times
        response_times = []
        success_count = 0
        
        for transaction in sample_transactions:
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.api_url}/predict",
                    json=transaction,
                    timeout=10
                )
                if response.status_code == 200:
                    success_count += 1
                response_time = time.time() - start_time
                response_times.append(response_time)
            except Exception as e:
                logger.error(f"Request failed: {str(e)}")
        
        # Calculate statistics
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            p95_response_time = np.percentile(response_times, 95)
            p99_response_time = np.percentile(response_times, 99)
            success_rate = success_count / n_samples
            
            logger.info(f"Response time statistics:")
            logger.info(f"  Average: {avg_response_time:.4f} seconds")
            logger.info(f"  P95: {p95_response_time:.4f} seconds")
            logger.info(f"  P99: {p99_response_time:.4f} seconds")
            logger.info(f"  Success rate: {success_rate:.4f}")
            
            # Visualize response times
            plt.figure(figsize=(10, 6))
            sns.histplot(response_times, kde=True)
            plt.title("API Response Time Distribution", fontsize=16)
            plt.xlabel("Response Time (seconds)", fontsize=14)
            plt.ylabel("Frequency", fontsize=14)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f"{self.output_dir}/response_times_{timestamp}.png", 
                      dpi=300, bbox_inches="tight")
            
            return {
                "avg_response_time": avg_response_time,
                "p95_response_time": p95_response_time,
                "p99_response_time": p99_response_time,
                "success_rate": success_rate
            }
        else:
            logger.error("No successful response times recorded")
            return None
    
    def check_for_model_drift(self):
        """Check for model drift using feedback data"""
        logger.info("Checking for model drift")
        
        # Load feedback data
        feedback_data = load_feedback_data()
        
        if feedback_data is None or len(feedback_data) < 100:
            logger.info("Not enough feedback data to check for model drift")
            return None
        
        # Calculate calibration error
        calibration_error = calculate_calibration_error(feedback_data)
        
        logger.info(f"Model calibration error: {calibration_error:.4f}")
        
        # Visualize calibration
        visualize_calibration(feedback_data)
        
        # Check if retraining is needed
        if calibration_error > MODEL_RETRAINING_THRESHOLD:
            logger.warning(f"Model drift detected with calibration error: {calibration_error:.4f}")
            logger.warning("Model retraining is recommended")
            return {
                "drift_detected": True,
                "calibration_error": calibration_error
            }
        else:
            logger.info("No significant model drift detected")
            return {
                "drift_detected": False,
                "calibration_error": calibration_error
            }
    
    def check_cloudwatch_logs(self, hours=24):
        """Check CloudWatch logs for errors"""
        logger.info(f"Checking CloudWatch logs for the past {hours} hours")
        
        try:
            # Calculate time range
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = int((datetime.now() - timedelta(hours=hours)).timestamp() * 1000)
            
            # Query CloudWatch logs
            response = self.cloudwatch_client.filter_log_events(
                logGroupName=CLOUDWATCH_LOG_GROUP,
                startTime=start_time,
                endTime=end_time,
                filterPattern="ERROR"
            )
            
            error_logs = response.get('events', [])
            
            logger.info(f"Found {len(error_logs)} error logs in the past {hours} hours")
            
            if error_logs:
                for log in error_logs[:10]:  # Show first 10 errors
                    logger.info(f"Error log: {log['message']}")
            
            return {
                "error_count": len(error_logs),
                "sample_errors": [log['message'] for log in error_logs[:10]] if error_logs else []
            }
        except Exception as e:
            logger.error(f"Failed to check CloudWatch logs: {str(e)}")
            return None
    
    def run_full_monitoring_check(self):
        """Run a full monitoring check"""
        logger.info("Starting full monitoring check")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "api_health": None,
            "response_times": None,
            "model_drift": None,
            "error_logs": None
        }
        
        # Check API health
        health_status, health_data = self.check_api_health()
        results["api_health"] = {
            "status": health_status,
            "details": health_data
        }
        
        # Monitor response times
        if health_status:
            response_time_stats = self.monitor_response_times(n_samples=100)
            results["response_times"] = response_time_stats
        
        # Check for model drift
        drift_results = self.check_for_model_drift()
        results["model_drift"] = drift_results
        
        # Check error logs
        error_logs = self.check_cloudwatch_logs(hours=24)
        results["error_logs"] = error_logs
        
        # Save monitoring results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"{self.output_dir}/monitoring_report_{timestamp}.json", "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Monitoring report saved to {self.output_dir}/monitoring_report_{timestamp}.json")
        
        # Trigger actions based on results
        self.act_on_monitoring_results(results)
        
        return results
    
    def act_on_monitoring_results(self, results):
        """Take actions based on monitoring results"""
        # Check if API is unhealthy
        if not results["api_health"]["status"]:
            logger.warning("API is unhealthy. Sending alert...")
            self.send_alert("API Health Alert", "Transaction prediction API is unhealthy!")
        
        # Check for slow response times
        if results["response_times"] and results["response_times"]["p95_response_time"] > 2.0:
            logger.warning("API response times are slow. Sending alert...")
            self.send_alert("Performance Alert", 
                           f"Slow API response times detected: P95 = {results['response_times']['p95_response_time']:.2f}s")
        
        # Check for model drift
        if results["model_drift"] and results["model_drift"]["drift_detected"]:
            logger.warning("Model drift detected. Initiating retraining...")
            self.trigger_model_retraining()
        
        # Check for high error rate
        if results["error_logs"] and results["error_logs"]["error_count"] > 10:
            logger.warning(f"High error rate detected: {results['error_logs']['error_count']} errors")
            self.send_alert("Error Rate Alert", 
                           f"High API error rate: {results['error_logs']['error_count']} errors in the last 24 hours")
    
    def send_alert(self, title, message):
        """Send an alert (in a real system, this would send to PagerDuty, Slack, email, etc.)"""
        logger.warning(f"ALERT - {title}: {message}")
        # In a real system, integrate with your alerting platform here
        
    def trigger_model_retraining(self):
        """Trigger model retraining process"""
        logger.info("Triggering model retraining process")
        # In a real system, this would kick off a model retraining job
        # For example, by calling an AWS Lambda function or starting an ECS task

def generate_random_transactions(n_samples=1000):
    """Generate random transactions for testing"""
    # Define possible values for categorical features
    payment_methods = ["credit_card", "debit_card", "bank_transfer", "mobile_payment"]
    device_types = ["mobile", "web", "pos", "tablet"]
    customer_locations = ["urban", "suburban", "rural"]
    transaction_results = ["success", "pending", "failed"]
    latency_bins = ["low", "medium", "high"]
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    amount_bins = ["low", "medium", "high"]

    # Generate random data
    num_transactions = n_samples
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
    transactions = pd.DataFrame(data)

    
    return transactions

def load_feedback_data():
    """Load feedback data from files"""
    feedback_files = [f for f in os.listdir(FEEDBACK_DIR) if f.startswith("feedback_data_")]
    
    if not feedback_files:
        return None
    
    all_feedback = []
    for file in feedback_files:
        try:
            df = pd.read_csv(os.path.join(FEEDBACK_DIR, file))
            all_feedback.append(df)
        except Exception as e:
            logger.error(f"Error loading feedback file {file}: {str(e)}")
    
    if all_feedback:
        return pd.concat(all_feedback, ignore_index=True)
    else:
        return None

def calculate_calibration_error(feedback_data):
    """Calculate model calibration error"""
    # Group by probability buckets
    feedback_data["prob_bucket"] = pd.cut(feedback_data["prediction_probability"], bins=10)
    
    # Calculate calibration error by bucket
    calibration_by_bucket = feedback_data.groupby("prob_bucket")["actual_success"].agg(["mean", "count"]).reset_index()
    
    # Calculate the overall calibration error (weighted average of absolute differences)
    total_samples = calibration_by_bucket["count"].sum()
    
    calibration_by_bucket["bucket_midpoint"] = calibration_by_bucket["prob_bucket"].apply(
        lambda x: (x.left + x.right) / 2
    )
    
    calibration_by_bucket["abs_error"] = abs(
        calibration_by_bucket["bucket_midpoint"] - calibration_by_bucket["mean"]
    )
    
    calibration_by_bucket["weighted_error"] = (
        calibration_by_bucket["abs_error"] * calibration_by_bucket["count"] / total_samples
    )
    
    total_calibration_error = calibration_by_bucket["weighted_error"].sum()
    
    return total_calibration_error

def visualize_calibration(feedback_data):
    """Visualize model calibration"""
    # Group by probability buckets
    feedback_data["prob_bucket"] = pd.cut(feedback_data["prediction_probability"], bins=10)
    
    # Calculate calibration by bucket
    calibration_by_bucket = feedback_data.groupby("prob_bucket")["actual_success"].agg(["mean", "count"]).reset_index()
    
    # Calculate bucket midpoints
    calibration_by_bucket["bucket_midpoint"] = calibration_by_bucket["prob_bucket"].apply(
        lambda x: (x.left + x.right) / 2
    )
    
    # Create calibration plot
    plt.figure(figsize=(10, 8))
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    
    # Actual calibration points
    plt.scatter(
        calibration_by_bucket["bucket_midpoint"], 
        calibration_by_bucket["mean"],
        s=calibration_by_bucket["count"] / 10,  # Size proportional to number of samples
        alpha=0.7
    )
    
    # Add error bars
    for _, row in calibration_by_bucket.iterrows():
        plt.annotate(
            f"{row['count']}",
            (row["bucket_midpoint"], row["mean"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center"
        )
    
    plt.title("Model Calibration Plot", fontsize=16)
    plt.xlabel("Predicted Probability", fontsize=14)
    plt.ylabel("Observed Frequency", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{MONITORING_OUTPUT_DIR}/calibration_plot_{timestamp}.png", 
              dpi=300, bbox_inches="tight")

# Example usage
if __name__ == "__main__":
    monitor = ModelMonitor()
    monitor.run_full_monitoring_check()