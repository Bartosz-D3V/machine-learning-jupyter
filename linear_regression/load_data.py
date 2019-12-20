import os

import pandas as pd
from pandas import DataFrame


def load_data(path: str, filename: str) -> DataFrame:
    csv_path = os.path.join(path, filename)
    return pd.read_csv(csv_path)
