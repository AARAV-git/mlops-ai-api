# import os
# os.environ["PYTHONIOENCODING"] = "utf-8"

# # ==============================
# # Azure ML + MLflow Connection
# # ==============================
# from azure.ai.ml import MLClient
# from azure.identity import DefaultAzureCredential
# import mlflow

# ml_client = MLClient(
#     DefaultAzureCredential(),
#     subscription_id="523ef521-5b9a-4bb8-a36f-073f8df68503",
#     resource_group_name="mlops-workshops3",
#     workspace_name="mlops_workshop3"
# )

# # Set MLflow tracking URI to Azure
# # mlflow.set_tracking_uri("azureml://centralindia.api.azureml.ms/mlflow/v1.0/subscriptions/523ef521-5b9a-4bb8-a36f-073f8df68503/resourceGroups/mlops-workshops3/providers/Microsoft.MachineLearningServices/workspaces/mlops_workshop3")

# # ==============================
# # ML Code
# # ==============================
# from sklearn.linear_model import LogisticRegression
# from sklearn.datasets import load_iris
# from sklearn.metrics import accuracy_score
# import joblib

# # Load dataset
# X, y = load_iris(return_X_y=True)

# # Set experiment
# mlflow.set_experiment("iris-exp")

# # ==============================
# # Training + Logging
# # ==============================
# with mlflow.start_run():

#     # Train model
#     model = LogisticRegression(max_iter=200)
#     model.fit(X, y)

#     # Predictions
#     preds = model.predict(X)
#     acc = accuracy_score(y, preds)

#     # Log parameters & metrics
#     mlflow.log_param("model", "LogisticRegression")
#     mlflow.log_param("max_iter", 200)
#     mlflow.log_metric("accuracy", acc)

#     # Save model locally
#     joblib.dump(model, "model.pkl")

#     # Log as artifact (IMPORTANT FIX)
#     mlflow.log_artifact("model.pkl")

#     print("Accuracy:", acc)



# ==============================Version 2: Local MLflow Tracking

# import mlflow
# import mlflow.sklearn
# from sklearn.datasets import make_classification
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# import pickle

# mlflow.set_tracking_uri("http://127.0.0.1:5000")

# with mlflow.start_run():

#     X, y = make_classification(n_samples=100, n_features=4)

#     model = RandomForestClassifier(n_estimators=100)
#     model.fit(X, y)

#     preds = model.predict(X)
#     acc = accuracy_score(y, preds)

#     print("Accuracy:", acc)

#     # 🔥 LOGGING
#     mlflow.log_metric("accuracy", acc)
#     mlflow.log_param("n_estimators", 100)

#     # save model
#     with open("model.pkl", "wb") as f:
#         pickle.dump(model, f)

#     # log model
#     mlflow.sklearn.log_model(model, "model")

    # =====================version 2: log as artifact

# import mlflow
# import mlflow.sklearn
# from sklearn.datasets import make_classification
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pickle

# mlflow.set_tracking_uri("http://127.0.0.1:5000")

# with mlflow.start_run():

#     # Data
#     X, y = make_classification(n_samples=200, n_features=4)

#     # Model
#     model = RandomForestClassifier(n_estimators=100)
#     model.fit(X, y)

#     # Predictions
#     preds = model.predict(X)
#     acc = accuracy_score(y, preds)

#     print("Accuracy:", acc)

#     # Log metrics
#     mlflow.log_metric("accuracy", acc)
#     mlflow.log_param("n_estimators", 100)

#     # 🔥 Confusion Matrix
#     cm = confusion_matrix(y, preds)

#     plt.figure()
#     sns.heatmap(cm, annot=True, fmt="d")
#     plt.title("Confusion Matrix")

#     plt.savefig("confusion_matrix.png")
#     mlflow.log_artifact("confusion_matrix.png")

#     # 🔥 Feature Importance Graph
#     importances = model.feature_importances_

#     plt.figure()
#     plt.bar(range(len(importances)), importances)
#     plt.title("Feature Importance")

#     plt.savefig("feature_importance.png")
#     mlflow.log_artifact("feature_importance.png")

#     # Save model
#     with open("model.pkl", "wb") as f:
#         pickle.dump(model, f)

#     # Log model
#     mlflow.sklearn.log_model(model, "model")

#=============================================Version 3: DVC Integration    
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

# ==============================
# MLflow Setup
# ==============================
import mlflow
import mlflow.sklearn

# 👉 Use LOCAL MLflow (recommended for now)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("iris-exp")

# ==============================
# ML Code
# ==============================
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load dataset
X, y = load_iris(return_X_y=True)

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

    print("Accuracy:", acc)

    # ==============================
    # Log Params & Metrics
    # ==============================
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", 200)
    mlflow.log_metric("accuracy", acc)

    # ==============================
    # Save Model
    # ==============================
    joblib.dump(model, "model.pkl")

    # Log model properly (better than artifact)
    mlflow.sklearn.log_model(model, "model")

    # ==============================
    # 🔥 Add Graphs (important)
    # ==============================

    # Confusion Matrix
    cm = confusion_matrix(y, preds)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # ==============================
    # Done
    # ==============================
    print("Run logged successfully in MLflow")