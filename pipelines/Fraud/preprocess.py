"""Feature engineers the Fraud dataset."""
import argparse
import logging
import os
import pathlib
import requests
import tempfile
import pickle
import tarfile
import sys

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

# Determine the directory of the current script
current_dir = os.path.dirname(os.path.realpath(__file__))
print(f"Current Directory {current_dir}")

sys.path.insert(0, current_dir)
print(f"Current Directory {current_dir}")

from custom_transformers import FrequencyEncoder, FeatureEngineeringTransformer

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
    pathlib.Path(f"{base_dir}/stream").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{base_dir}/onhold").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{base_dir}/artifacts").mkdir(parents=True, exist_ok=True)

    input_data = args.input_data

    print(input_data)
        
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

    logger.info("Transform X_test")
    X_test_transformed = preprocessing_pipeline.transform(X_test)

    logger.info("Transform X_validation")
    X_validation_transformed = preprocessing_pipeline.transform(X_validation)

    logger.info("Concatenating X and y for both train, test and Validation")
    process_train_set_with_pipeline = pd.concat([y_train, X_train_transformed],axis=1)
    process_test_set_with_pipeline = pd.concat([y_test, X_test_transformed],axis=1)
    process_validation_set_with_pipeline = pd.concat([y_validation, X_validation_transformed],axis=1)

    # Write CSV files
    process_train_set_with_pipeline.to_csv(f"{base_dir}/train/train.csv", index=False)
    logger.info("Train dataset written to %s/train/train.csv", base_dir)
    process_test_set_with_pipeline.to_csv(f"{base_dir}/test/test.csv", index=False, header=False)
    logger.info("Test dataset written to %s/test/test.csv", base_dir)
    process_validation_set_with_pipeline.to_csv(f"{base_dir}/validation/validation.csv", index=False)
    logger.info("Validation dataset written to %s/validation/validation.csv", base_dir)
    stream.to_csv(f"{base_dir}/stream/stream.csv", index=False)
    logger.info("Stream dataset written to %s/stream/stream.csv", base_dir)
    onhold.to_csv(f"{base_dir}/onhold/onhold.csv", index=False)
    logger.info("Onhold dataset written to %s/onhold/onhold.csv", base_dir)

    # Ensure the artifacts directory exists and is clean
    artifacts_dir = os.path.join(base_dir, "artifacts")
    pathlib.Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Artifacts directory created at %s", artifacts_dir)

    # Clear any existing preprocess.tar.gz to avoid directory conflict
    model_tar = os.path.join(artifacts_dir, "preprocess.tar.gz")
    if os.path.exists(model_tar):
        if os.path.isdir(model_tar):
            shutil.rmtree(model_tar)  # Remove if it’s a directory
        else:
            os.remove(model_tar)  # Remove if it’s a file
        logger.info("Cleared existing %s", model_tar)

    # Export the preprocessor.pkl file
    logger.info("Exporting model.joblib file")
    preprocessor_path = os.path.join(artifacts_dir, "model.joblib")
    with open(preprocessor_path, "wb") as f:
        joblib.dump(preprocessing_pipeline, f)
    logger.info("model.joblib file saved successfully to %s", preprocessor_path)

    # Package the model.joblib into a tar.gz archive
    if os.path.exists(preprocessor_path):
        with tarfile.open(model_tar, "w:gz") as tar:
            tar.add(preprocessor_path, arcname="model.joblib")
        logger.info("Packaged model.joblib into %s", model_tar)
    else:
        logger.error("model.joblib does not exist at %s", preprocessor_path)
        raise FileNotFoundError(f"model.joblibl not found at {preprocessor_path}")

    # Verify tar file creation
    if os.path.exists(model_tar):
        logger.info("Tar file %s created successfully with size %d bytes", model_tar, os.path.getsize(model_tar))
    else:
        logger.error("Failed to create tar file at %s", model_tar)
        raise FileNotFoundError(f"Tar file not created at {model_tar}")