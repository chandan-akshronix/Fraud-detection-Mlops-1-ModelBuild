"""Evaluation script for measuring classification metrics."""
import json
import logging
import pathlib
import pickle
import tarfile

import numpy as np
import pandas as pd
import xgboost

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.debug("Starting evaluation.")
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    logger.debug("Loading xgboost model.")
    model = pickle.load(open("xgboost-model", "rb"))

    logger.debug("Reading test data.")
    test_path = "/opt/ml/processing/test/test.csv"
    df = pd.read_csv(test_path, header=None)

    logger.debug("Preparing test data.")
    # First column (index 0) is the label, rest are features
    y_test = df.iloc[:, 0].to_numpy()  # Label is column 0
    X_test = df.iloc[:, 1:]  # Features are all columns after 0
    X_test_dmatrix = xgboost.DMatrix(X_test.values)

    logger.info("Performing predictions against test data.")
    predictions = model.predict(X_test)
    predictions_binary = (predictions > 0.5).astype(int)  # Threshold for binary classification

    logger.debug("Calculating classification metrics.")
    precision = precision_score(y_test, predictions_binary)
    recall = recall_score(y_test, predictions_binary)
    f1 = f1_score(y_test, predictions_binary)
    auc = roc_auc_score(y_test, predictions)

    report_dict = {
        "classification_metrics": {
            "precision": {"value": precision},
            "recall": {"value": recall},
            "f1_score": {"value": f1},
            "auc_roc": {"value": auc},
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Writing out evaluation report with metrics: precision=%f, recall=%f, f1=%f, auc=%f",
                precision, recall, f1, auc)
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))