from dependencies import * 
# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Define helper functions for reusability
def plot_missing_values(df):
    """
    Plot missing values in the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    """
    print(f"Missing values count:\n{df.isnull().sum()}")
    print(f"\nMissing values percentage:\n{df.isnull().sum() / len(df) * 100:.2f}%")
    
    plt.figure(figsize=(12, 6))
    msno.matrix(df)
    plt.title('Missing Values Matrix')
    plt.show()
    
    plt.figure(figsize=(12, 6))
    msno.bar(df)
    plt.title('Missing Values Bar Chart')
    plt.show()

def detect_outliers(df, column, method='iqr', threshold=1.5):
    """
    Detect outliers in a given column using specified method.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    column : str
        Column name to check for outliers
    method : str, optional (default='iqr')
        Method to use for outlier detection ('iqr' or 'zscore')
    threshold : float, optional (default=1.5)
        Threshold for outlier detection
        
    Returns:
    --------
    outlier_mask : pandas.Series
        Boolean mask of outliers
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    elif method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        outlier_mask = z_scores > threshold
    else:
        raise ValueError("Method must be either 'iqr' or 'zscore'")
    
    return outlier_mask

def plot_outliers(df, column, method='iqr', threshold=1.5):
    """
    Plot outliers in a given column.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    column : str
        Column name to check for outliers
    method : str, optional (default='iqr')
        Method to use for outlier detection ('iqr' or 'zscore')
    threshold : float, optional (default=1.5)
        Threshold for outlier detection
    """
    outlier_mask = detect_outliers(df, column, method, threshold)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df[column])
    plt.title(f'Boxplot of {column}')
    
    plt.subplot(1, 2, 2)
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.tight_layout()
    plt.show()
    
    print(f"Number of outliers detected: {outlier_mask.sum()}")
    print(f"Percentage of outliers: {outlier_mask.sum() / len(df) * 100:.2f}%")

def plot_correlation_matrix(df, figsize=(20, 16)):
    """
    Plot correlation matrix for numerical features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    figsize : tuple, optional (default=(20, 16))
        Figure size
    """
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    return corr