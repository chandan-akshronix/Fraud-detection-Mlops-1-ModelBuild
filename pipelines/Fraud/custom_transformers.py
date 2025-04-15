import tempfile
import pickle
import tarfile

import boto3
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import set_config # output in pandas dataframe of pipeline

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.freq_dict = {}
    def fit(self, X, y=None):
        # X can be a DataFrame or array; convert to Series if single column
        if isinstance(X, pd.DataFrame):
            for col in X.columns:
                self.freq_dict[col] = X[col].value_counts().to_dict()
        else:
            self.freq_dict['col'] = pd.Series(X).value_counts().to_dict()
        return self
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
            for col in X.columns:
                X_transformed[col] = X[col].map(self.freq_dict.get(col, {})).fillna(0)
            return X_transformed
        else:
            return pd.Series(X).map(self.freq_dict.get('col', {})).fillna(0).values

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler_amount_zscore = StandardScaler()
        self.shipping_address_freq = {}
        # These are computed from training data
        self.high_amount_quantile = None 
        self.high_quantity_quantile = None
    def fit(self, X, y=None):
        X = X.copy()
        # Fit scaler for 'Transaction Amount' to create 'Amount_zscore' and calculating high quartile value
        if 'Transaction Amount' in X.columns:
            self.scaler_amount_zscore.fit(X[['Transaction Amount']])
            self.high_amount_quantile = np.percentile(X['Transaction Amount'], 95)
        # High quantity quartile value
        if 'Quantity' in X.columns:
             self.high_quantity_quantile = np.percentile(X['Quantity'], 95)
        # Compute frequency mapping for 'Shipping Address'
        if 'Shipping Address' in X.columns:
            self.shipping_address_freq = X['Shipping Address'].value_counts().to_dict()
        return self
    def transform(self, X):
        X = X.copy()
        ### Handle Missing Values
        numeric_cols = ['Transaction Amount', 'Quantity', 'Customer Age', 'Account Age Days', 'Transaction Hour']
        for col in numeric_cols:
            if col in X.columns:
                X[col] = X[col].fillna(X[col].median())
        categorical_cols = ['Payment Method', 'Product Category', 'Customer Location', 'Device Used']
        for col in categorical_cols:
            if col in X.columns:
                X[col] = X[col].fillna('Unknown')
        ### Transaction Amount Features
        if 'Transaction Amount' in X.columns:
            X['Amount_Log'] = np.log1p(X['Transaction Amount'])
            X['Amount_zscore'] = self.scaler_amount_zscore.transform(X[['Transaction Amount']])
        if 'Quantity' in X.columns and 'Transaction Amount' in X.columns:
            X['Amount_per_Quantity'] = X['Transaction Amount'] / (X['Quantity'] + 1)
        ### Date Features
        if 'Transaction Date' in X.columns:
            X['Transaction Date'] = pd.to_datetime(X['Transaction Date'])
            X['Is_Weekend'] = X['Transaction Date'].dt.dayofweek.isin([5, 6]).astype(int)
            X['Day_of_Week'] = X['Transaction Date'].dt.dayofweek
            X['Month'] = X['Transaction Date'].dt.month
            X['Day_of_Year'] = X['Transaction Date'].dt.dayofyear
            X['Is_Month_Start'] = X['Transaction Date'].dt.is_month_start.astype(int)
            X['Is_Month_End'] = X['Transaction Date'].dt.is_month_end.astype(int)
        if 'Transaction Hour' in X.columns:
            X['Hour_Bin'] = pd.cut(X['Transaction Hour'], bins=[-np.inf, 6, 12, 18, np.inf], 
                                  labels=['Night', 'Morning', 'Afternoon', 'Evening'])
            X['hour_sin'] = np.sin(2 * np.pi * X['Transaction Hour'] / 24)
            X['hour_cos'] = np.cos(2 * np.pi * X['Transaction Hour'] / 24)
            X['Unusual_Hour_Flag'] = ((X['Transaction Hour'] < 6) | (X['Transaction Hour'] > 22)).astype(int)
        if 'Day_of_Week' in X.columns:
            X['weekday_sin'] = np.sin(2 * np.pi * X['Day_of_Week'] / 7)
            X['weekday_cos'] = np.cos(2 * np.pi * X['Day_of_Week'] / 7)
        if 'Month' in X.columns:
            X['month_sin'] = np.sin(2 * np.pi * X['Month'] / 12)
            X['month_cos'] = np.cos(2 * np.pi * X['Month'] / 12)
        ### Customer Profile Features
        if 'Customer Age' in X.columns:
            X['Age_Category'] = pd.cut(X['Customer Age'], bins=[-np.inf, 0, 25, 35, 50, 65, np.inf], 
                                      labels=['Invalid', 'Young', 'Young_Adult', 'Adult', 'Senior', 'Elder'])
        if 'Account Age Days' in X.columns:
            X['Account_Age_Weeks'] = X['Account Age Days'] // 7
            X['Is_New_Account'] = (X['Account Age Days'] <= 30).astype(int)
        ### Transaction Size Bins
        if 'Transaction Amount' in X.columns:
            bin_edges = [0, 55.51, 114.44, 197.74, 343.42, np.inf]  # Adjust based on your data
            bin_labels = ['Very_Small', 'Small', 'Medium', 'Large', 'Very_Large']
            X['Transaction_Size'] = pd.cut(X['Transaction Amount'], bins=bin_edges, labels=bin_labels)
        if 'Quantity' in X.columns:
            X['Quantity_Log'] = np.log1p(X['Quantity'])
            X['High_Quantity_Flag'] = (X['Quantity'] > self.high_quantity_quantile).astype(int)
        ### Location and Device Features
        if 'Customer Location' in X.columns and 'Device Used' in X.columns:
            X['Location_Device'] = X['Customer Location'] + '_' + X['Device Used']
        ### Address Features
        for col in ['Shipping Address', 'Billing Address']:
            if col in X.columns:
                X[col] = X[col].str.lower().str.replace(r'[^\w\s]', '', regex=True).str.strip()
        if 'Shipping Address' in X.columns and 'Billing Address' in X.columns:
            X['Address_Match'] = (X['Shipping Address'] == X['Billing Address']).astype(int)
        if 'Shipping Address' in X.columns:
            X['Shipping Address Frequency'] = X['Shipping Address'].map(self.shipping_address_freq).fillna(0)
        ### Risk Indicators
        if 'Transaction Amount' in X.columns:
            X['High_Amount_Flag'] = (X['Transaction Amount'] > self.high_amount_quantile).astype(int)
        ### Drop Unnecessary Columns
        columns_to_drop = ['Customer ID', 'Transaction ID', 'Transaction Date', 'IP Address', 'Shipping Address', 'Billing Address']
        X = X.drop(columns=[col for col in columns_to_drop if col in X.columns])
        
        return X