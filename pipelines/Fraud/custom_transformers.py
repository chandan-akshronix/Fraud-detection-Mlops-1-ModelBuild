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
        pass  # No parameters to initialize for now
    
    def fit(self, X, y=None):
        # Since this transformer doesn't need to learn from the data (e.g., compute statistics),
        # we just return self. Add fitting logic here if needed in the future.
        return self
    
    def transform(self, X):
        X = X.copy()
        
        ### Handle Missing Values
        numeric_cols = ['Transaction Amount', 'Quantity', 'Customer Age', 'Account Age Days', 'Transaction Hour']
        for col in numeric_cols:
            if col in X.columns:
                X[col] = X[col].fillna(X[col].median())
        
        categorical_cols = ['Customer Location']
        for col in categorical_cols:
            if col in X.columns:
                X[col] = X[col].fillna('Unknown')
        
        ### Transaction Amount Features
        if 'Transaction Amount' in X.columns:
            X['Amount_Log'] = np.log1p(X['Transaction Amount'])
        if 'Quantity' in X.columns and 'Transaction Amount' in X.columns:
            X['Amount_per_Quantity'] = X['Transaction Amount'] / (X['Quantity'] + 1)
        
        ### Date Features
        if 'Transaction Date' in X.columns:
            X['Transaction Date'] = pd.to_datetime(X['Transaction Date'])
            X['Day_of_Week'] = X['Transaction Date'].dt.dayofweek
            X['Day_of_Year'] = X['Transaction Date'].dt.dayofyear

        
        if 'Transaction Hour' in X.columns:
            X['hour_sin'] = np.sin(2 * np.pi * X['Transaction Hour'] / 24)
            X['hour_cos'] = np.cos(2 * np.pi * X['Transaction Hour'] / 24)

        if 'Account Age Days' in X.columns:
            X['Account_Age_Weeks'] = X['Account Age Days'] // 7
        
        ### Location and Device Features
        if 'Customer Location' in X.columns and 'Device Used' in X.columns:
            X['Location_Device'] = X['Customer Location'] + '_' + X['Device Used']        
        
        ### Drop Unnecessary Columns
        columns_to_drop = ['Customer ID', 'Transaction ID', 'Transaction Date', 'IP Address', 'Shipping Address', 'Billing Address',"Payment Method","Product Category","Device Used"]
        X = X.drop(columns=[col for col in columns_to_drop if col in X.columns])
        
        return X