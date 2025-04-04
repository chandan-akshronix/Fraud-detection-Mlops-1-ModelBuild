import json
import logging
import pathlib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.debug("Starting post-transform evaluation.")
    
    # Path to the predictions file from the batch transform step
    predictions_path = "/opt/ml/processing/predictions/test.csv.out"
    
    # Read the predictions file (no headers)
    logger.debug("Reading predictions from %s", predictions_path)
    df = pd.read_csv(predictions_path, header=None)
    
    # Extract prediction probabilities (column 0) and true labels (column 1)
    logger.debug("Extracting predictions and true labels.")
    y_pred_prob = df.iloc[:, 0].to_numpy()  # First column: prediction probabilities
    y_true = df.iloc[:, 1].to_numpy()       # Second column: true labels
    
    # Convert probabilities to binary predictions using a 0.5 threshold
    logger.debug("Converting probabilities to binary predictions.")
    predictions_binary = (y_pred_prob > 0.5).astype(int)
    
    # Calculate classification metrics
    logger.debug("Calculating classification metrics.")
    precision = precision_score(y_true, predictions_binary)
    recall = recall_score(y_true, predictions_binary)
    f1 = f1_score(y_true, predictions_binary)
    auc = roc_auc_score(y_true, y_pred_prob)  # AUC uses probabilities, not binary predictions
    
    # Create the report dictionary
    report_dict = {
        "classification_metrics": {
            "precision": {"value": precision},
            "recall": {"value": recall},
            "f1_score": {"value": f1},
            "auc_roc": {"value": auc},
        },
    }
    
    # Define output directory and file path
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    evaluation_path = f"{output_dir}/evaluation.json"
    
    # Write the evaluation report
    logger.info("Writing evaluation report with metrics: precision=%f, recall=%f, f1=%f, auc=%f",
                precision, recall, f1, auc)
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
    
    logger.info("Evaluation report written to %s", evaluation_path)