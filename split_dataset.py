from typing import Tuple
from sklearn.model_selection import train_test_split

import pandas as pd


def split_dataset(
    data: pd.DataFrame, class_: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X: pd.DataFrame = data.drop(labels=class_, axis=1)
    y: pd.DataFrame = data[class_]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test
