import json
import sys
from pathlib import Path

import mlflow
import mlflow.shap
import mlflow.sklearn
import pandas as pd
from mlflow.models import evaluate, infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.config import DATA_PROCESSED, MLFLOW_TRACKING_URI

sys.path.append(str(Path(__file__).parent.parent.parent))

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

final_dataset = pd.read_parquet(DATA_PROCESSED / "final_dataset_dr.parquet")

with open(DATA_PROCESSED / "selected_features.json", encoding="utf-8") as f:
    features = json.load(f)


X_train = final_dataset[final_dataset["q_num"] < -7][features]
X_test = final_dataset[final_dataset["q_num"] >= -7][features]
y_train = final_dataset[final_dataset["q_num"] < -7]["default_probability"]
y_test = final_dataset[final_dataset["q_num"] >= -7]["default_probability"]


preprocessor = ColumnTransformer(
    transformers=[("num", StandardScaler(), X_train.columns)],
    remainder="drop",
    n_jobs=-1,
)

params_Lasso = {
    "alpha": 0.0001,
    "fit_intercept": True,
    "random_state": 42,
}
model = Lasso(**params_Lasso)
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train_and_log(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    RUN_NAME,
    EXPERIMENT_NAME,
    REGISTRY_MODEL_NAME,
    metadata,
):
    """
    Обучает модель, вычисляет метрики, логирует в mlflow и вычисляет метрики.
    """
    # Запускаем обучение модели
    model.fit(X_train, y_train)

    eval_data = X_test.copy()
    eval_data["label"] = y_test

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
    else:
        experiment_id = experiment.experiment_id

    with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id):
        params = {k: str(v) for k, v in model.named_steps["model"].get_params().items()}
        mlflow.log_params(params)
        # Предсказания
        y_pred = model.predict(X_test)
        signature = infer_signature(X_test, y_pred)
        input_example = X_test[:10]
        pip_requirements = "requirements.txt"

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="dr_pipeline",
            registered_model_name=REGISTRY_MODEL_NAME,
            metadata=metadata,
            signature=signature,
            input_example=input_example,
            pip_requirements=pip_requirements,
            await_registration_for=60,
        )
        result = evaluate(
            model_info.model_uri,
            eval_data,
            targets="label",
            model_type="regressor",
            evaluator_config={"log_explainer": True},
        )

        for artifact_name in result.artifacts:
            if "shap" in artifact_name.lower():
                print(f"Generated: {artifact_name}")


metadata = {
    "model_type": "regression",
    "model_class": str(model),
    "features": str(features),
    "notes": "baseline model, with feature selection yet",
    "status": "experimental",
}
EXPERIMENT_NAME = "dr_macro_regression"
RUN_NAME = "baseline_all_features"
REGISTRY_MODEL_NAME = "dr_macro_model"


train_and_log(
    model=pipeline,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    RUN_NAME=RUN_NAME,
    EXPERIMENT_NAME=EXPERIMENT_NAME,
    REGISTRY_MODEL_NAME=REGISTRY_MODEL_NAME,
    metadata=metadata,
)
