from setuptools import setup, find_packages

setup(
    name="transaction_success_optimization",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "flask",
        "fastapi",
        "uvicorn",
        "streamlit"
    ],
)
