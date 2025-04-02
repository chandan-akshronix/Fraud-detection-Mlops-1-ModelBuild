import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn.metrics import roc_auc_score

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.debug("Starting cross-validation.")
    train_path = "/opt/ml/processing/train/train.csv"
    df = pd.read_csv(train_path)
    y = df.iloc[:, 0].to_numpy()  # First column is label
    X = df.iloc[:, 1:].to_numpy()

    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=108)
    best_auc = 0
    best_model = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        params = {
            "objective": "binary:logistic",
            "max_depth": 5,
            "eta": 0.2,
            "gamma": 4,
            "min_child_weight": 6,
            "subsample": 0.7,
            "scale_pos_weight": float(np.sum(y_train == 0)) / np.sum(y_train == 1),
            "silent": 0,
        }

        model = xgb.train(params, dtrain, num_boost_round=50, evals=[(dval, "validation")])
        predictions = model.predict(dval)
        auc = roc_auc_score(y_val, predictions)

        logger.info(f"Fold {fold} AUC: {auc}")
        if auc > best_auc:
            best_auc = auc
            best_model = model

    output_dir = "/opt/ml/processing/cv_model"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "xgboost-model"), "wb") as f:
        pickle.dump(best_model, f)
    logger.info(f"Best model saved with AUC: {best_auc}")