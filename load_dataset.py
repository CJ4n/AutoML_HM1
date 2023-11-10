import openml
from pandas import DataFrame


def load_dataset_from_id(id: int) -> DataFrame:
    return openml.datasets.get_dataset(dataset_id=id).get_data(
        dataset_format="dataframe"
    )[0]
