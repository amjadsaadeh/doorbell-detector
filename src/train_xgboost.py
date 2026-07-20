import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

MLFLOW_EXPERIMENT_NAME = "doorbell-detector"


def prepare_data(df):
    # Convert MFCC features to 1D arrays
    X = np.vstack([x.flatten() for x in df["mfcc_features"]])
    # Convert labels to binary (background=0, non-background=1)
    le = LabelEncoder()
    # Convert to inary problem
    df["label"] = df["label"].apply(
        lambda x: "background" if x == "background" else "bell"
    )
    y = le.fit_transform(df["label"])
    return X, y, le


def main():

    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    model_params = params["model"]

    # Load data
    df = pd.read_hdf("./data/balanced_data.h5", key="data")
    X, y, le = prepare_data(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["training"]["test_size"], random_state=42, stratify=y
    )

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    mlflow.xgboost.autolog(log_datasets=False)

    with mlflow.start_run():
        mlflow.log_param("test_size", params["training"]["test_size"])

        # Train model
        model = xgb.XGBClassifier(objective="binary:logistic", **model_params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

        # Evaluate and save metrics
        y_pred = model.predict(X_test)
        y_pred_decoded = le.inverse_transform(y_pred)
        y_test_decoded = le.inverse_transform(y_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        mlflow.log_metric("f1_score", report["weighted avg"]["f1-score"])
        mlflow.log_metric("recall", report["weighted avg"]["recall"])
        mlflow.log_metric("precision", report["weighted avg"]["precision"])

        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test_decoded, y_pred_decoded, ax=ax)
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        # Save model
        model_path = Path("./models/xgboost_model.json")
        model_path.parent.mkdir(exist_ok=True)
        model.save_model(model_path)


if __name__ == "__main__":
    main()
