from pathlib import Path

import pandas as pd

META_COLUMNS = [
    "_id",
    "branchname",
    "brandId",
    "brandname",
    "coordinates.latitude",
    "coordinates.longitude",
    "country",
    "settings.takeawayOnly",
]

META_DTYPES = {
    "_id": pd.StringDtype(),
    "branchname": pd.StringDtype(),
    "brandId": pd.StringDtype(),
    "brandname": pd.StringDtype(),
    "coordinates.latitude": pd.Float64Dtype(),
    "coordinates.longitude": pd.Float64Dtype(),
    "country": pd.StringDtype(),
}

SALES_DTYPES = {"unique_id": pd.StringDtype(), "y": pd.Float64Dtype()}

SALES_DATE_COLUMNS = ["ds"]


def read_sales_from_disk(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(file_path, dtype=SALES_DTYPES, parse_dates=SALES_DATE_COLUMNS)


def read_meta_from_disk(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(file_path, usecols=META_COLUMNS, dtype=META_DTYPES)
