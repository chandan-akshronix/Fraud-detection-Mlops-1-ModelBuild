"""Feature engineers the Fraud dataset."""
import argparse
import logging
import os
import pathlib
import joblib
import tarfile
import boto3
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import set_config
from collections import Counter
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from custom_transformers import FrequencyEncoder, FeatureEngineeringTransformer

# Determine the directory of the current script
current_dir = os.path.dirname(os.path.realpath(__file__))

print(f"Current Directory {current_dir}")

print("Files in current directory:", os.listdir('/opt/ml/processing/input/code'))


# Initialize logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def apply_resampling(X_train, y_train):
    smote = BorderlineSMOTE(sampling_strategy=0.10, random_state=108, k_neighbors=5, m_neighbors=10)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    logger.info(f"After SMOTE: {Counter(y_res)}")
    undersampler = RandomUnderSampler(sampling_strategy=0.33, random_state=108, replacement=True)
    X_res, y_res = undersampler.fit_resample(X_res, y_res)
    logger.info(f"After UnderSampling: {Counter(y_res)}")
    return X_res, y_res

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

    logger.info(f"Main Dataframe column names before anything {df.columns}")

    df = df.drop(columns=["Unnamed: 0"], axis = 1)

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
        high_cardinality_cols = ['Customer Location', 'Location_Device']

        # Encoding transformer
        encoding_transformer = ColumnTransformer(
            transformers=[
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

    logger.info(f"X_train column names before transform {X_train.columns} & 1st Row of X_train is {X_train.head(1)}")

    X_train_transformed = preprocessing_pipeline.fit_transform(X_train)

    logger.info(f"X_train_transformed column names before transform {X_train_transformed.columns} & 1st Row of X_train is {X_train_transformed.head(1)}")

    logger.info("Transform X_test")

    logger.info(f"X_test column names before transform {X_test.columns} & 1st Row of X_train is {X_test.head(1)}")

    X_test_transformed = preprocessing_pipeline.transform(X_test)

    logger.info(f"X_test_transformed column names before transform {X_test_transformed.columns} & 1st Row of X_train is {X_test_transformed.head(1)}")

    logger.info("Transform X_validation")
    X_validation_transformed = preprocessing_pipeline.transform(X_validation)

    X_resampled, y_resampled = apply_resampling(X_train_transformed, y_train)

    logger.info("Concatenating X and y for both train, test and Validation")
    process_train_set_with_pipeline = pd.concat([y_resampled, X_resampled],axis=1)
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