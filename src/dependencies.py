
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

import numpy as np
# For reproducibility
np.random.seed(42)
# For plotting
import matplotlib.pyplot as plt
import seaborn as sns
# For better visualizations
plt.style.use('ggplot')
sns.set(style="whitegrid")
import missingno as msno
from datetime import datetime
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine learning libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import shap
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

