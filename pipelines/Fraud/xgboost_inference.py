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

            # Clean the input: remove newlines and extra spaces
            cleaned_input = request_body.replace('\n', '').replace('\r', '').strip()

            # Split by commas and convert to floats
            data = np.array([list(map(float, cleaned_input.split(',')))])
            dmatrix = xgb.DMatrix(data)
            logger.info(f"Input data shape: {data.shape}")
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

def output_fn(prediction, accept):
    """Format the prediction output."""
    logger.info(f"Prediction type: {type(prediction)}, value: {prediction}")
    probability = prediction[0]
    threshold = 0.5
    binary_prediction = 1 if probability > threshold else 0
    logger.info(f"Probability: {probability}, Binary prediction: {binary_prediction}")
    
    if accept == 'text/csv':
        return f"{probability},{binary_prediction}", 'text/csv'
    elif accept == 'application/json':
        return json.dumps({"probability": probability, "prediction": binary_prediction}), 'application/json'
    else:
        raise ValueError(f"Unsupported accept type: {accept}")