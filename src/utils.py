from src.dependencies import * 
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
    print(f"\nMissing values percentage:\n{(df.isnull().sum() / len(df) * 100).round(2)}%")  # FIXED

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

# Define a function to evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # Fit the model
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Print metrics
    print(f"\n{model_name} Performance:")
    print(f"Training time: {train_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Create and plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Failure', 'Success'],
                yticklabels=['Failure', 'Success'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()
    
    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()
    
    # Plot Precision-Recall curve
    plt.figure(figsize=(8, 6))
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.plot(recall_curve, precision_curve, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.grid(alpha=0.3)
    plt.show()
    
    # Return results for comparison
    return {
        'model': model,
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'train_time': train_time,
        'y_pred_proba': y_pred_proba
    }
    
    
    
def create_nn_model(input_dim, activation=None, optimizer=None, loss=None, metrics=None, output_activation=None):
    model = Sequential([
        Dense(64, activation=activation, input_dim=input_dim),
        Dropout(0.3),
        Dense(32, activation=activation),
        Dropout(0.2),
        Dense(16, activation=activation),
        Dense(1, output_activation=output_activation)
    ])
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model
