import numpy as np
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def detect_outliers(df, column, method='iqr', threshold=1.5):
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        return (df[column] < (Q1 - threshold * IQR)) | (df[column] > (Q3 + threshold * IQR))
    return np.zeros(df.shape[0], dtype=bool)

def feature_engineering_pipeline(df):
    # Fill missing values
    fill_values = {'network_latency': df['network_latency'].median(), 'payment_amount': df['payment_amount'].median()}
    df.fillna(fill_values, inplace=True)
    
    # Outlier handling
    if 'payment_amount' in df.columns:
        df['amount_log'] = np.log1p(df['payment_amount'])
    
    if 'network_latency' in df.columns:
        outliers_mask = detect_outliers(df, 'network_latency')
        latency_cap = df['network_latency'].quantile(0.99)
        df.loc[outliers_mask, 'network_latency'] = latency_cap
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Categorical processing
    categorical_cols = ['payment_method', 'customer_location', 'device_type', 'time_of_day', 'latency_bin', 'day_name']
    ordinal_cols = ['amount_bin', 'latency_bin']
    ordinal_mappings = [['very_low', 'low', 'medium', 'high', 'very_high'], ['low', 'medium', 'high', 'very_high']]
    
    # Feature transformations
    numeric_features = ['amount_log', 'network_latency', 'merchant_rolling_avg_amount', 'hourly_transaction_volume', 'merchant_success_rate']
    num_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    
    cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    ord_transformer = Pipeline(steps=[('ordinal', OrdinalEncoder(categories=ordinal_mappings))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numeric_features),
            ('cat', cat_transformer, categorical_cols),
            ('ord', ord_transformer, ordinal_cols)
        ]
    )
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    
    return pipeline

# Usage with a dataset
df = pd.read_csv("data/raw/transactions_march_2023.csv") 
pipeline = feature_engineering_pipeline(df)

# Save pipeline
with open('models/feature_engineering_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
