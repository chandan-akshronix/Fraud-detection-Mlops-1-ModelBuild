"""Feature engineers the Fraud dataset."""
import argparse
import logging
import os
import pathlib
import requests
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

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    # Create directories for train, validation, test outputs
    pathlib.Path(f"{base_dir}/train").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{base_dir}/validation").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{base_dir}/test").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)

    input_data = args.input_data
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    s3_path = f"{base_dir}/data/input.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, s3_path)

    logger.info("Reading downloaded data.")
    df = pd.read_csv(s3_path)

    logger.info("First Row of Dataframe")
    print(df.head(1))

    os.unlink(s3_path)

    logger.info("Splitting Main dataset for Training")
    main, stream_onhold = train_test_split(df, test_size=0.10, stratify=df["Is Fraudulent"], random_state=108)
    print(f"Main Set shape {main.shape}")

    logger.info("Splitting Main dataset into main_train, main_test_validation")
    train, test_validation = train_test_split(main, test_size=0.30, stratify=main["Is Fraudulent"], random_state=108)
    print(f"Train Set shape {train.shape}")

    logger.info("Splitting Main dataset into main_train, main_test_validation")
    test, validation = train_test_split(test_validation, test_size=0.50, stratify=test_validation["Is Fraudulent"], random_state=108)
    print(f"Test Set shape {test.shape}")
    print(f"Validation Set shape {validation.shape}")

    logger.info("Splitting stream and onHold set")
    stream, onhold = train_test_split(stream_onhold, test_size=0.3, stratify=stream_onhold["Is Fraudulent"], random_state=108)
    print(f"Stream Set shape {stream.shape}")
    print(f"Onhold Set shape {onhold.shape}")

    logger.info("Splitting the Main dataset in X and y versions of Train, Test and Validation")
    
    X_train = train.drop(["Is Fraudulent"], axis=1)
    y_train = train["Is Fraudulent"]

    X_test = test.drop(["Is Fraudulent"], axis=1)
    y_test = test["Is Fraudulent"]

    X_validation = validation.drop(["Is Fraudulent"], axis=1)
    y_validation = validation["Is Fraudulent"]

    
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

    def create_preprocessing_pipeline():
        # Categorical columns after feature engineering
        low_cardinality_cols = ['Payment Method', 'Product Category', 'Device Used', 
                               'Hour_Bin', 'Age_Category', 'Transaction_Size']
        high_cardinality_cols = ['Customer Location', 'Location_Device']

        # Encoding transformer
        encoding_transformer = ColumnTransformer(
            transformers=[
                ('onehot', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'), 
                 low_cardinality_cols),
                ('freq', FrequencyEncoder(), high_cardinality_cols)
            ],
            remainder='passthrough'  # Pass through numerical columns
        )

        # Full preprocessing pipeline
        preprocessing_pipeline = Pipeline([
            ('feature_engineering', FeatureEngineeringTransformer()),
            ('encoding', encoding_transformer),
            ('scaling', StandardScaler())
        ])

        return preprocessing_pipeline
    
    logger.info("Preprocessing pipeline initiated")

    preprocessing_pipeline = create_preprocessing_pipeline()

    set_config(transform_output="pandas") # default o/p is numpy array, but to avoid explanability issues with models, need o/p in DataFrame

    logger.info("Fit and Transform X_train")
    X_train_transformed = preprocessing_pipeline.fit_transform(X_train)

    logger.info("Fit and Transform X_test")
    X_test_transformed = preprocessing_pipeline.transform(X_test)

    logger.info("Fit and Transform X_validation")
    X_validation_transformed = preprocessing_pipeline.transform(X_validation)

    logger.info("Concatenating X and y for both train, test and Validation")
    process_train_set_with_pipeline = pd.concat([X_train_transformed,y_train],axis=1)
    process_test_set_with_pipeline = pd.concat([X_test_transformed,y_test],axis=1)
    process_validation_set_with_pipeline = pd.concat([X_validation_transformed,y_validation],axis=1)


    logger.info("Writing out train, test, stream and onhold datasets to %s.", base_dir)
    process_train_set_with_pipeline.to_csv(f"{base_dir}/train/train.csv", index=False)
    process_test_set_with_pipeline.to_csv(f"{base_dir}/test/test.csv", index=False)
    process_validation_set_with_pipeline.to_csv(f"{base_dir}/validation/validation.csv", index=False)
    stream.to_csv(f"{base_dir}/data/stream.csv", index=False)
    onhold.to_csv(f"{base_dir}/data/onhold.csv", index=False)

    # Ensure the artifacts directory exists
    artifacts_dir = os.path.join(base_dir, "artifacts")
    pathlib.Path(artifacts_dir).mkdir(parents=True, exist_ok=True)

    # Export the preprocessor.pkl file
    logger.info("Exporting preprocessor.pkl file")
    with open(os.path.join(artifacts_dir, "preprocessor.pkl"), "wb") as f:
        pickle.dump(preprocessing_pipeline, f)
    logger.info("preprocessor.pkl file saved successfully.")

    # Package the preprocessor.pkl into a tar.gz archive
    model_tar = os.path.join(artifacts_dir, "model.tar.gz")
    preprocessor_path = os.path.join(artifacts_dir, "preprocessor.pkl")
    
    if os.path.exists(preprocessor_path):
        with tarfile.open(model_tar, "w:gz") as tar:
            tar.add(preprocessor_path, arcname="preprocessor.pkl")
        logger.info(f"Packaged preprocessor.pkl into {model_tar}")
    else:
        logger.error(f"preprocessor.pkl does not exist at {preprocessor_path}")
        raise FileNotFoundError(f"preprocessor.pkl not found at {preprocessor_path}")