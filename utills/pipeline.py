from typing import List, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline


def calculate_mse(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series):
    # Ensure X_test and y_test are the correct types
    if not isinstance(X_test, (pd.DataFrame, np.ndarray)):
        raise ValueError("X_test must be a pandas DataFrame or numpy array")
    if not isinstance(y_test, (pd.Series, np.ndarray)):
        raise ValueError("y_test must be a pandas Series or numpy array")

    # Generating predictions and calculating MSE
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")


def evaluate_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
):
    if not isinstance(X_val, (pd.DataFrame, np.ndarray)):
        raise ValueError("X_test must be a pandas DataFrame or numpy array")
    if not isinstance(y_val, (pd.Series, np.ndarray)):
        raise ValueError("y_test must be a pandas Series or numpy array")

    pipeline.fit(X, y)

    test_score = pipeline.score(X_val, y_val)
    train_score = pipeline.score(X, y)
    print("Parameter set: " + str(pipeline.named_steps["model"]))
    print("Test score R^2: " + str(test_score))
    print("Train score R^2: " + str(train_score))
    calculate_mse(pipeline, X_val, y_val)


def evaluate_pipeline_on_datasets(
    pipeline: Pipeline, optimal_config, datasets: List[Tuple[DataFrame, Series]]
):
    for X, y in datasets:
        pipeline.set_params(**optimal_config)
        evaluate_pipeline(
            pipeline=pipeline,
            X=X,
            y=y,
            X_val=X,
            y_val=y,
        )

def get_column_transformer() -> ColumnTransformer:
    num_pipeline = Pipeline(
        steps=[("impute", SimpleImputer(strategy="mean")), ("scale", MinMaxScaler())]
    )
    cat_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("one-hot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    col_trans = ColumnTransformer(
        transformers=[
            (
                "num_pipeline",
                num_pipeline,
                make_column_selector(dtype_include=np.number),
            ),
            ("cat_pipeline", cat_pipeline, make_column_selector(dtype_include=object)),
        ],
        remainder="drop",
        n_jobs=-1,
    )
    return col_trans