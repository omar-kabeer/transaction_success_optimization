# Create model_retraining.py with the following content

# model_retraining.py content
import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("retraining.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
FEEDBACK_DIR = "./feedback"
MODEL_DIR = "./models"
TRAINING_DATA_PATH = "./data/training_data.csv"
MODEL_PATH = os.path.join(MODEL_DIR, "Optimized XGBoost.pkl.pkl")
PIPELINE_PATH = os.path.join(MODEL_DIR, "feature_engineering_pipeline.pkl")

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)

def load_feedback_data():
    """Load feedback data for retraining"""
    logger.info("Loading feedback data for retraining")
    
    feedback_files = [f for f in os.listdir(FEEDBACK_DIR) if f.startswith("feedback_data_")]
    
    if not feedback_files:
        logger.warning("No feedback data found")
        return None
    
    all_feedback = []
    for file in feedback_files:
        try:
            df = pd.read_csv(os.path.join(FEEDBACK_DIR, file))
            all_feedback.append(df)
        except Exception as e:
            logger.error(f"Error loading feedback file {file}: {str(e)}")
    
    if all_feedback:
        feedback_df = pd.concat(all_feedback, ignore_index=True)
        logger.info(f"Loaded {len(feedback_df)} feedback records")
        return feedback_df
    else:
        return None

def load_training_data():
    """Load original training data"""
    logger.info(f"Loading original training data from {TRAINING_DATA_PATH}")
    
    try:
        training_df = pd.read_csv(TRAINING_DATA_PATH)
        logger.info(f"Loaded {len(training_df)} training records")
        return training_df
    except Exception as e:
        logger.error(f"Error loading training data: {str(e)}")
        return None

def combine_training_and_feedback_data(training_df, feedback_df):
    """Combine original training data with feedback data"""
    logger.info("Combining training and feedback data")
    
    # Process feedback data to match training data format
    feedback_df = feedback_df.rename(columns={"is_successful": "result_numeric"})
    
    # Here we're assuming the feedback data contains the necessary features
    
    # Combined enriched dataset
    combined_df = pd.concat([training_df, feedback_df], ignore_index=True)
    logger.info(f"Combined dataset contains {len(combined_df)} records")
    
    return combined_df

def retrain_model(training_data):
    """Retrain the model with updated data"""
    logger.info("Retraining model with updated data")
    
    # Split data
    X = training_data.drop(["result_numeric", "transaction_id"], axis=1, errors="ignore")
    y = training_data["result_numeric"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load preprocessing pipeline
    preprocessing_pipeline = joblib.load(PIPELINE_PATH)
    
    # Transform features
    X_train_processed = preprocessing_pipeline.transform(X_train)
    X_test_processed = preprocessing_pipeline.transform(X_test)
    
    # Initialize and train model
    logger.info("Training XGBoost model")
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42
    )
    
    model.fit(
        X_train_processed, 
        y_train,
        eval_set=[(X_test_processed, y_test)],
        verbose=True
    )
    
    # Evaluate model
    y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
    y_pred = model.predict(X_test_processed)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    logger.info(f"Model performance metrics:")
    logger.info(f"  AUC: {auc:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    
    return model, {
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def save_model(model, metrics):
    """Save the retrained model"""
    logger.info("Saving retrained model")
    
    # Create versioned filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_model_path = os.path.join(MODEL_DIR, f"xgboost_model_{timestamp}.pkl")
    
    # Save model
    joblib.dump(model, versioned_model_path)
    
    # Also overwrite the main model file
    joblib.dump(model, MODEL_PATH)
    
    # Save metrics
    metrics_path = os.path.join(MODEL_DIR, f"model_metrics_{timestamp}.json")
    import json
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Model saved to {versioned_model_path}")
    logger.info(f"Model metrics saved to {metrics_path}")
    
    return versioned_model_path, metrics_path

def run_retraining_pipeline():
    """Run the full model retraining pipeline"""
    logger.info("Starting model retraining pipeline")
    
    # Load feedback data
    feedback_data = load_feedback_data()
    if feedback_data is None:
        logger.error("Cannot retrain model: No feedback data available")
        return False
    
    # Load original training data
    training_data = load_training_data()
    if training_data is None:
        logger.error("Cannot retrain model: No training data available")
        return False
    
    # Combine data
    combined_data = combine_training_and_feedback_data(training_data, feedback_data)
    
    # Retrain model
    model, metrics = retrain_model(combined_data)
    
    # Save model
    model_path, metrics_path = save_model(model, metrics)
    
    logger.info("Model retraining completed successfully")
    
    # Archive processed feedback files
    archive_feedback_files()
    
    return True

def archive_feedback_files():
    """Archive feedback files that have been used for retraining"""
    logger.info("Archiving processed feedback files")
    
    archive_dir = os.path.join(FEEDBACK_DIR, "archived")
    os.makedirs(archive_dir, exist_ok=True)
    
    feedback_files = [f for f in os.listdir(FEEDBACK_DIR) if f.startswith("feedback_data_")]
    
    for file in feedback_files:
        try:
            os.rename(
                os.path.join(FEEDBACK_DIR, file),
                os.path.join(archive_dir, file)
            )
        except Exception as e:
            logger.error(f"Error archiving feedback file {file}: {str(e)}")
    
    logger.info(f"Archived {len(feedback_files)} feedback files")

# Example usage
if __name__ == "__main__":
    success = run_retraining_pipeline()
    if success:
        logger.info("Model retraining pipeline completed successfully")
    else:
        logger.error("Model retraining pipeline failed")