# ==========================
# General Libraries
# ==========================
import os
import time
import uuid
import random
import warnings
import joblib
import pickle
import datetime
import numpy as np
import pandas as pd
from collections import defaultdict
from faker import Faker
from scipy import stats
from datetime import datetime, timedelta
import logging
import requests
import json
import concurrent.futures
from collections import Counter
from collections import defaultdict

# ==========================
# Visualization Libraries
# ==========================
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# Styling for better visualization
plt.style.use('ggplot')
sns.set(style="whitegrid")

# ==========================
# Machine Learning Libraries
# ==========================
from sklearn.model_selection import (train_test_split, cross_val_score, 
                                     KFold, StratifiedKFold, GridSearchCV)
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, 
                                   OneHotEncoder, LabelEncoder)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, precision_recall_curve, 
                             precision_score, recall_score, f1_score, 
                             roc_auc_score, roc_curve, auc, 
                             classification_report, accuracy_score,
                             precision_recall_fscore_support)

# ==========================
# Time Series & Statistical Analysis
# ==========================
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
# Multi-Armed Bandit implementation
from scipy import stats

# ==========================
# Advanced Machine Learning
# ==========================
import xgboost as xgb
import optuna  # Hyperparameter tuning

# ==========================
# Neural Networks
# ==========================
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm   


# ==========================
# Reinforcement Learning
# ==========================

import gym
from gym import spaces
import stable_baselines3
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env


# ==========================
# Model Interpretability
# ==========================
import shap

# ==========================
# Warnings Configuration
# ==========================
from sklearn.exceptions import ConvergenceWarning
from tqdm import TqdmWarning

# Suppress TqdmWarning
warnings.simplefilter("ignore", category=TqdmWarning)

# Suppress warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================
# Pandas Configuration
# ==========================
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# ==========================
# FastAPI Libraries
# ==========================
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
import boto3