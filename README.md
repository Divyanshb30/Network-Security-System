# Network Intrusion Detection System (End-to-End ML + MLOps Project)

An end-to-end Machine Learning pipeline for detecting network intrusions using real-world MLOps techniques. This project goes beyond model training — it includes a complete training pipeline, modular architecture, data validation with drift detection, MLflow integration via DagsHub, and a FastAPI-based deployment layer.

## Key Features
- Modular pipeline using classes & configs
- Imputation with KNNImputer
- SMOTE for class balancing
- Model selection with GridSearchCV
- Metrics: F1, Precision, Recall
- MLflow + DagsHub tracking
- Drift detection on test/train split
- FastAPI for serving model predictions

## Tech Stack & Tools

-  **Machine Learning**: Scikit-learn (Random Forest, KNN, Logistic Regression, Gradient Boosting, AdaBoost)
-  **Data Preprocessing**: KNNImputer, SMOTE (oversampling), Feature Pipelines
-  **Experiment Tracking**: MLflow with [DagsHub Integration](https://dagshub.com)
-  **Model Serving**: FastAPI (with HTML predictions)
-  **Drift Monitoring**: KS Test

----


