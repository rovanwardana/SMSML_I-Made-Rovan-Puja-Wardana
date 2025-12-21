
import mlflow
import mlflow.xgboost 
import dagshub
import json
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import os
import warnings
warnings.filterwarnings("ignore")
dagshub.init(
    repo_owner="rovanwardana",
    repo_name="diabetes-mlflow",
    mlflow=True
)
mlflow.set_experiment("Diabetes_XGBoost_Tuning_Advanced")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "diabetes_dataset_preprocessing")

TRAIN_PATH = os.path.join(DATASET_DIR, "diabetes_train.csv")
TEST_PATH = os.path.join(DATASET_DIR, "diabetes_test.csv")

TARGET_COL = "diabetes"

train_df = pd.read_csv(TRAIN_PATH).dropna(subset=[TARGET_COL])
test_df = pd.read_csv(TEST_PATH).dropna(subset=[TARGET_COL])

X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL]
X_test = test_df.drop(columns=[TARGET_COL])
y_test = test_df[TARGET_COL]

model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    use_label_encoder=False,
    tree_method='hist'
)

param_dist = {
    "n_estimators": [100, 150, 200, 250, 300, 350, 400],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0, 0.1, 0.2, 0.3],
    "reg_alpha": [0, 0.01, 0.1, 0.5, 1.0],
    "reg_lambda": [1, 1.5, 2, 3]
}


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=100, 
    scoring="accuracy", 
    cv=skf,
    random_state=42,
    n_jobs=-1,
    verbose=2,
    return_train_score=True
)

mlflow.set_tracking_uri("https://dagshub.com/rovanwardana/diabetes-mlflow.mlflow")

with mlflow.start_run(run_name="XGBoost_Diabetes_Tuning"):
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    best_params = search.best_params_

    y_pred = best_model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

    mlflow.log_params(best_params)
    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    try:
        mlflow.xgboost.log_model(
            xgb_model=best_model,
            artifact_path="model"
        )
        print("Model logged successfully")
    except Exception as e:
        print(f"Error logging model: {e}")
        # Fallback: Save as pickle
        import pickle
        with open("model.pkl", "wb") as f:
            pickle.dump(best_model, f)
        mlflow.log_artifact("model.pkl")
        print("Model logged as pickle artifact")

    #artifact confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    cm_path = os.path.join(BASE_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)

    # artifact classification report
    cls_report = classification_report(y_test, y_pred, output_dict=True)

    cls_path = os.path.join(BASE_DIR, "classification_report.json")
    with open(cls_path, "w") as f:
        json.dump(cls_report, f, indent=4)

    mlflow.log_artifact(cls_path)

    # artifact feature importance
    importances = best_model.feature_importances_

    plt.figure(figsize=(8, 4))
    plt.bar(X_train.columns, importances)
    plt.xticks(rotation=90)
    plt.title("Feature Importance")
    plt.tight_layout()
    fi_path = os.path.join(BASE_DIR, "feature_importance_tuning.png")
    plt.savefig(fi_path)
    plt.close()
    mlflow.log_artifact(fi_path)

    # artifact metrics summary
    metrics_path = os.path.join(BASE_DIR, "metrics_summary.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    mlflow.log_artifact(metrics_path)

    print("TRAINING & LOGGING SELESAI")

