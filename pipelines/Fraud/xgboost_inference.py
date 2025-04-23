import json
import xgboost as xgb
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def model_fn(model_dir):
    try:
        model = xgb.Booster()
        model_path = os.path.join(model_dir, 'xgboost-model')  # Load the specific file
        logger.info(f"Loading XGBoost model from {model_path}")
        logger.info(f"Files in model directory: {os.listdir(model_dir)}")
        model.load_model(model_path)
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def input_fn(request_body, request_content_type):
    """Parse input data from the request."""
    if request_content_type == 'text/csv':
        try:
            # Log the raw input for debugging
            logger.info(f"Raw input data: {request_body}")
            
            # Split the request body by newlines to handle multiple records
            lines = request_body.strip().split('\n')
            data = []
            for line in lines:
                if line.strip():  # Skip empty lines
                    row = list(map(float, line.strip().split(',')))
                    data.append(row)
            if not data:
                raise ValueError("No data in request")
            # Create DMatrix with multiple rows
            dmatrix = xgb.DMatrix(np.array(data))
            logger.info(f"Input data shape: {np.array(data).shape}")
            return dmatrix
        except Exception as e:
            logger.error(f"Error parsing CSV input: {str(e)}")
            raise ValueError(f"Invalid CSV input: {str(e)}")
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make predictions using the XGBoost model."""
    logger.info("Starting prediction")
    try:
        probabilities = model.predict(input_data)
        logger.info(f"Prediction probabilities: {probabilities}")
        return probabilities
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

def output_fn(predictions, accept):
    """Format the prediction output for multiple records."""
    logger.info(f"Predictions: {predictions}")
    threshold = 0.5
    if accept == 'text/csv':
        output_lines = []
        for probability in predictions:
            binary_prediction = 1 if probability > threshold else 0
            output_lines.append(f"{probability},{binary_prediction}")
        return '\n'.join(output_lines), 'text/csv'
    elif accept == 'application/json':
        results = []
        for probability in predictions:
            binary_prediction = 1 if probability > threshold else 0
            results.append({"probability": float(probability), "prediction": binary_prediction})
        return json.dumps(results), 'application/json'
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
