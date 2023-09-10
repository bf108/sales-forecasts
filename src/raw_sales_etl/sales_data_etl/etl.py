from datetime import date
from pathlib import Path
from typing import Union

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


def merge_sales_meta(
    df_sales: pd.DataFrame,
    sales_unique_id: str,
    df_meta: pd.DataFrame,
    meta_unique_id: str,
) -> pd.DataFrame:
    df_comb = (
        pd.merge(
            df_sales,
            df_meta,
            how="inner",
            left_on=sales_unique_id,
            right_on=meta_unique_id,
        )
        .drop(columns=[meta_unique_id])
        .rename(columns={"ds": "date"})
        .set_index("date")
    )
    df_comb.loc[:, "country"] = df_comb.country.str.lower()
    df_comb.loc[:, "operational_flag"] = True
    return df_comb


def get_unique_business_ids(
    df_sales: pd.DataFrame, unique_id: Union[str, None] = None
) -> list[str]:
    unique_id = unique_id if unique_id else "unique_id"
    return list(df_sales[unique_id].unique())


def zero_negative_sales(dfs: pd.DataFrame, sales_col: str) -> pd.DataFrame:
    dfs.loc[dfs[sales_col] < 0, f"{sales_col}_adj"] = 0
    return dfs


def create_calendar_df(start_dt: date, end_dt: date) -> pd.DataFrame:
    df_calendar = pd.DataFrame(pd.date_range(start_dt, end_dt, freq="d")).rename(
        columns={0: "date"}
    )
    df_calendar.loc[:, "dummy"] = "calendar"
    df_calendar.set_index("date", inplace=True)
    return df_calendar


def join_calendar_to_sales_history(
    df_sales: pd.DataFrame, df_calendar: pd.DataFrame
) -> pd.DataFrame:
    return df_calendar.join(df_sales, how="left").drop(columns=["dummy"])


def join_hols_to_sales_history_calendar(
    df_sales: pd.DataFrame, df_hols: pd.DataFrame
) -> pd.DataFrame:
    df_output = (
        df_sales.set_index("country", append=True)
        .join(df_hols.set_index("country", append=True), how="left")
        .reset_index()
        .set_index("date")
    )
    return df_output


def determine_day(holiday_flag: bool, lead_up_flag: bool) -> str:
    if holiday_flag:
        return "holiday"
    if lead_up_flag:
        return "lead_up"
    return "normal"


def create_lead_up_columns(df_input: pd.DataFrame) -> pd.DataFrame:
    df_input.loc[df_input["flag_holiday"].isna(), "flag_holiday"] = False
    df_input["flag_lead_up_holiday"] = df_input["flag_holiday"].shift(periods=-1)
    df_input["lead_up_holiday_name"] = df_input["holiday_name"].shift(periods=-1)
    df_input["day_type"] = df_input.apply(
        lambda x: determine_day(x["flag_holiday"], x["flag_lead_up_holiday"]), axis=1
    )
    return df_input


def get_last_5_weeks_sales_per_day(
    df_sales: pd.DataFrame, sales_col: str
) -> pd.DataFrame:
    df_sales["sales_7_days_prior"] = df_sales[sales_col].shift(7)
    df_sales["sales_14_days_prior"] = df_sales[sales_col].shift(14)
    df_sales["sales_21_days_prior"] = df_sales[sales_col].shift(21)
    df_sales["sales_28_days_prior"] = df_sales[sales_col].shift(28)
    df_sales["sales_35_days_prior"] = df_sales[sales_col].shift(35)
    return df_sales


def flag_14_out_of_28_days_sales_history(
    df_sales: pd.DataFrame, sales_col: str
) -> pd.DataFrame:
    df_sales["fc_14_in_28_days"] = (
        ~df_sales[sales_col].rolling(window=28, min_periods=14).sum().isna()
    )
    return df_sales
