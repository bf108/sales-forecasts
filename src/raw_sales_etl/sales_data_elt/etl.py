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
