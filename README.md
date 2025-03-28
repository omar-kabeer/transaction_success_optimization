# Transaction Success Rate Optimization 🚀

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## 📌 Project Overview

This project implements an advanced transaction success rate optimization system using **machine learning**, **reinforcement learning**, and **real-time inference** to dynamically route and improve financial transaction outcomes. The solution analyzes transaction patterns, predicts success probabilities, and implements optimal routing strategies to maximize success rates.

## ✨ Key Features

- **Intelligent Transaction Routing**: Predicts transaction success probability and routes accordingly
- **Real-time Inference API**: FastAPI-based endpoint for integrating with payment systems
- **Advanced ML Models**: XGBoost, Neural Networks, and Reinforcement Learning approaches
- **Feature Engineering Pipeline**: Comprehensive preprocessing for transaction data
- **Model Retraining**: Continuous learning from new transaction outcomes
- **Performance Monitoring**: Dashboards for tracking model performance and business metrics
- **AWS Deployment**: Containerized deployment to AWS ECS
- **A/B Testing**: Evaluation framework for comparing routing strategies

## 🗂️ Project Structure

```
transaction_success_optimization/
├── app/                     # API and deployment files
│   ├── app.py               # FastAPI application
│   ├── deploy_to_aws.py     # AWS ECS deployment script
│   ├── monitoring.py        # API and model monitoring
│   └── static/              # Static assets for the API
│       ├── favicon.ico      # Favicon for the API
│       └── site.webmanifest # Web app manifest
├── configs/                 # Configuration files
│   └── config.yaml          # Main configuration parameters
├── dashboards/              # Interactive analysis dashboards
│   └── home.py              # Main Streamlit dashboard
├── data/                    # Transaction datasets
│   ├── batches/             # Batch prediction datasets
│   │   ├── holiday_week.parquet  # Holiday period transactions
│   │   └── normal_week.csv       # Normal period transactions
│   ├── processed/           # Processed feature data
│   │   ├── full_feature_engineered_data.csv  # Data with all features
│   │   └── processed_transaction_data.csv    # Cleaned transaction data
│   └── raw/                 # Original transaction data
│       ├── batches/         # Raw batch data
│       └── transactions_march_2023.csv  # Raw transaction data
├── models/                  # Trained models and pipelines
│   ├── feature_engineering_pipeline.pkl  # Feature preprocessing pipeline
│   ├── model_metadata.json              # Model performance metrics
│   ├── Optimized XGBoost_model_20250320.joblib  # Optimized model
│   ├── Optimized XGBoost.pkl           # Latest optimized model
│   ├── rl_agent.pth                    # Reinforcement learning agent
│   └── XGBoost_model_20250320.joblib   # Base XGBoost model
├── notebooks/               # Development and analysis notebooks
│   ├── 01_data_exploration.ipynb       # Initial data analysis
│   ├── 02_feature_engineering.ipynb    # Feature development notebook
│   ├── 03_model_training_and_evaluation.ipynb  # Model training notebook
│   ├── 05_deployment_and_integration.ipynb     # Deployment notebook
│   └── imgs/                # Notebook images and diagrams
├── reports/                 # Generated reports and evaluations
│   └── overview.pdf         # Project overview document
├── src/                     # Source code
│   ├── data_generator.py    # Synthetic data generation
│   ├── dependencies.py      # Project dependencies
│   ├── feature_engineering.py # Feature creation and transformation
│   ├── model_retraining.py  # Continuous model improvement
│   └── utils.py             # Utility functions
├── tests/                   # Test suite
│   ├── ab_testing.py        # A/B testing framework
│   └── test_app.py          # API tests
├── requirements.txt         # Project dependencies
├── setup.py                 # Package installation configuration
└── README.md                # Project documentation
```

## ⚙️ Installation and Setup

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/omar-kabeer/transaction_success_optimization.git
cd transaction_success_optimization

# Install with base dependencies
pip install -r requirements.txt
```

### Advanced Installation Options

Install specific dependency groups:

```bash
# For machine learning components only
pip install -e ".[ml]"

# For development (includes testing tools)
pip install -e ".[dev]"

# For all dependencies
pip install -e ".[all]"
```

## 🚀 Running the Project

### Data Preprocessing

```bash
# Generate synthetic data (if needed)
python -m src.data_generator

# Process transaction data
python -m src.data_preprocessing
```

### Model Training

```bash
# Train the XGBoost model
python -m src.model_training --model xgboost

# Train the reinforcement learning model
python -m src.model_training --model rl
```

### API Deployment

```bash
# Run the FastAPI application locally
cd app
uvicorn app:app --reload --port 8000
```

### Dashboards

```bash
# Launch the model performance dashboard
streamlit run dashboards/model_performance_dashboard.py
```

## 📊 Monitoring and Evaluation

The project includes several monitoring components:

1. **API Health Monitoring**: Real-time checks of endpoint availability and response times
2. **Model Drift Detection**: Identifies when model performance degrades
3. **Transaction Analysis**: Visual dashboards for business metrics
4. **A/B Testing**: Framework for comparing routing strategies

## 💡 Model Architecture

The system employs multiple models:

- **XGBoost Classifier**: Primary model for success prediction
- **Reinforcement Learning Agent**: Dynamic optimization of routing decisions
- **Feature Engineering Pipeline**: Automated transformation of raw transaction data

## 📈 Performance Metrics

The system is evaluated on:

- Transaction success rate improvement
- Precision and recall for fraud detection
- Average response time for real-time predictions
- Business impact metrics (revenue, customer satisfaction)

## 🔄 Continuous Improvement

The system implements:

- Feedback loop for model retraining
- Monitoring of prediction vs. actual outcomes
- Concept drift detection
- Automated model retraining when performance degrades

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📬 Contact

For any questions or feedback, please reach out to [uksaid12@gmail.com](mailto:uksaid12@gmail.com).
