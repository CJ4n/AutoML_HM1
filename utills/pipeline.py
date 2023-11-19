from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from skopt import BayesSearchCV


# def calculate_mse(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series):
    # # Ensure X_test and y_test are the correct types
    # if not isinstance(X_test, (pd.DataFrame, np.ndarray)):
    #     raise ValueError("X_test must be a pandas DataFrame or numpy array")
    # if not isinstance(y_test, (pd.Series, np.ndarray)):
    #     raise ValueError("y_test must be a pandas Series or numpy array")

    # # Generating predictions and calculating MSE
    # predictions = model.predict(X_test)
    # mse = mean_squared_error(y_test, predictions)
    # print(f"Mean Squared Error: {mse}")


# def evaluate_pipeline(
#     pipeline: Pipeline,
#     X_train: pd.DataFrame,
#     y_train: pd.Series,
#     X_test: pd.DataFrame,
#     y_test: pd.Series,
# ) -> Tuple[float, float]:
#     if not isinstance(X_test, (pd.DataFrame, np.ndarray)):
#         raise ValueError("X_test must be a pandas DataFrame or numpy array")
#     if not isinstance(y_test, (pd.Series, np.ndarray)):
#         raise ValueError("y_test must be a pandas Series or numpy array")

#     pipeline.fit(X_train, y_train)

#     test_score = pipeline.score(X_test, y_test)
#     train_score = pipeline.score(X_train, y_train)
#     print("Parameter set: " + str(pipeline.named_steps["model"]))
#     print("Test score R^2: " + str(test_score))
#     print("Train score R^2: " + str(train_score))
#     calculate_mse(pipeline, X_test, y_test)
#     return test_score, train_score



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


def get_bayes_model(
    pipeline: Pipeline,
    search_space: Dict[str, Any],
    n_iter=50,
) -> BayesSearchCV:
    return BayesSearchCV(
        pipeline,
        # [(space, # of evaluations)]
        search_spaces=search_space,
        n_iter=n_iter,
        n_jobs=-1,
        cv=5,  # Set cv=None to disable cross-validation
    )
