import os
os.environ["PYTHONIOENCODING"] = "utf-8"

# ==============================
# Azure ML + MLflow Connection
# ==============================
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import mlflow

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="523ef521-5b9a-4bb8-a36f-073f8df68503",
    resource_group_name="mlops-workshops3",
    workspace_name="mlops_workshop3"
)

# Set MLflow tracking URI to Azure
# mlflow.set_tracking_uri("azureml://centralindia.api.azureml.ms/mlflow/v1.0/subscriptions/523ef521-5b9a-4bb8-a36f-073f8df68503/resourceGroups/mlops-workshops3/providers/Microsoft.MachineLearningServices/workspaces/mlops_workshop3")

# ==============================
# ML Code
# ==============================
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
X, y = load_iris(return_X_y=True)

# Set experiment
mlflow.set_experiment("iris-exp")

# ==============================
# Training + Logging
# ==============================
with mlflow.start_run():

    # Train model
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    # Predictions
    preds = model.predict(X)
    acc = accuracy_score(y, preds)

    # Log parameters & metrics
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", 200)
    mlflow.log_metric("accuracy", acc)

    # Save model locally
    joblib.dump(model, "model.pkl")

    # Log as artifact (IMPORTANT FIX)
    mlflow.log_artifact("model.pkl")

    print("Accuracy:", acc)