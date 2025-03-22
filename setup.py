from setuptools import setup, find_packages

# Define groups of dependencies
base_dependencies = [
    "pandas>=1.3.0",
    "numpy>=1.20.0",
    "scipy>=1.7.0",
    "joblib>=1.0.0",
    "requests>=2.25.0",
    "python-dateutil>=2.8.0",
]

ml_dependencies = [
    "scikit-learn>=1.0.0",
    "xgboost>=1.5.0",
    "imbalanced-learn>=0.8.0",
    "optuna>=2.10.0",
    "statsmodels>=0.13.0",
]

viz_dependencies = [
    "matplotlib>=3.4.0",
    "seaborn>=0.11.0",
    "missingno>=0.5.0",
]

api_dependencies = [
    "fastapi>=0.70.0",
    "uvicorn>=0.15.0",
    "pydantic>=1.8.0",
]

web_dependencies = [
    "streamlit>=1.0.0",
    "flask>=2.0.0",
]

dev_dependencies = [
    "pytest>=6.2.0",
    "black>=21.12b0",
    "flake8>=4.0.0",
    "isort>=5.10.0",
]

interpretability_dependencies = [
    "shap>=0.40.0",
]

# Combine all dependencies for the full list
all_dependencies = (
    base_dependencies 
    + ml_dependencies 
    + viz_dependencies 
    + api_dependencies 
    + web_dependencies 
    + dev_dependencies
    + interpretability_dependencies
)

setup(
    name="transaction_success_optimization",
    version="0.1.0",
    description="A machine learning system for optimizing transaction success rates",
    author="Transaction Success Team",
    author_email="uksaid12@gmail.com",
    url="https://github.com/omar-kabeer/transaction_success_optimization",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=base_dependencies,
    extras_require={
        "ml": ml_dependencies,
        "viz": viz_dependencies,
        "api": api_dependencies,
        "web": web_dependencies,
        "dev": dev_dependencies,
        "interpret": interpretability_dependencies,
        "all": all_dependencies,
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
    ],
    entry_points={
        "console_scripts": [
            "transaction-api=app.api:main",
            "model-training=app.model_training:main",
        ],
    },
)
