from datetime import date
from pathlib import Path
from typing import Union

import numpy as np
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
    df_comb["y"] = df_comb["y"].astype("float")
    return df_comb


def get_unique_business_ids(
    df_sales: pd.DataFrame, unique_id: Union[str, None] = None
) -> list[str]:
    unique_id = unique_id if unique_id else "unique_id"
    return list(df_sales[unique_id].unique())


def zero_negative_sales(df_input: pd.DataFrame, sales_col: str) -> pd.DataFrame:
    df_output = df_input.copy()
    df_output.loc[:, f"{sales_col}_adj"] = df_output[sales_col]
    df_output.loc[df_output[sales_col] < 0, f"{sales_col}_adj"] = 0
    df_output[f"{sales_col}_adj"] = df_output[f"{sales_col}_adj"].astype("float")
    df_output[f"{sales_col}_adj"] = df_output[f"{sales_col}_adj"] + 0.01
    return df_output


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
    df_output = df_calendar.join(df_sales, how="left").drop(columns=["dummy"])
    df_output["business_id"] = df_sales["unique_id"].unique()[0]
    df_output["operational_flag"] = df_output["operational_flag"].ffill()
    return df_output


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
    df_output = df_input.copy()
    df_output.loc[df_output["flag_holiday"].isna(), "flag_holiday"] = False
    df_output["flag_lead_up_holiday"] = df_output["flag_holiday"].shift(periods=-1)
    df_output["lead_up_holiday_name"] = df_output["holiday_name"].shift(periods=-1)
    df_output["day_type"] = df_output.apply(
        lambda x: determine_day(x["flag_holiday"], x["flag_lead_up_holiday"]), axis=1
    )
    return df_output


def get_last_5_weeks_sales_per_day(
    df_input: pd.DataFrame, sales_col: str
) -> pd.DataFrame:
    df_output = df_input.copy()
    df_output["sales_7_days_prior"] = df_output[sales_col].shift(7)
    df_output["sales_14_days_prior"] = df_output[sales_col].shift(14)
    df_output["sales_21_days_prior"] = df_output[sales_col].shift(21)
    df_output["sales_28_days_prior"] = df_output[sales_col].shift(28)
    df_output["sales_35_days_prior"] = df_output[sales_col].shift(35)
    return df_output


def flag_14_out_of_28_days_sales_history(
    df_input: pd.DataFrame, sales_col: str
) -> pd.DataFrame:
    df_output = df_input.copy()
    df_output["fc_14_in_28_days"] = (
        ~df_output[sales_col].rolling(window=28, min_periods=14).sum().isna()
    )
    return df_output


def create_7_day_forecast_columns(df_input: pd.DataFrame) -> pd.DataFrame:
    df_output = df_input.copy()
    df_output["7_day_forecast_prelim"] = df_output[
        [
            "sales_7_days_prior",
            "sales_14_days_prior",
            "sales_21_days_prior",
            "sales_28_days_prior",
        ]
    ].mean(axis=1)

    df_output["7_day_forecast"] = df_output["7_day_forecast_prelim"]
    # Set forecast to nan when less than 14 days of sales recorded in last 28 days
    df_output.loc[df_output["fc_14_in_28_days"] == False, "7_day_forecast"] = np.nan
    return df_output


def create_14_day_forecast_columns(df_input: pd.DataFrame) -> pd.DataFrame:
    df_output = df_input.copy()
    df_output["input_14_day_forecast"] = df_output[
        [
            "sales_14_days_prior",
            "sales_21_days_prior",
            "sales_28_days_prior",
            "sales_35_days_prior",
        ]
    ].mean(axis=1)

    df_output["14_day_forecast_prelim"] = df_output[
        [
            "input_14_day_forecast",
            "sales_14_days_prior",
            "sales_21_days_prior",
            "sales_28_days_prior",
        ]
    ].mean(axis=1)

    # Set forecast to nan when less than 14 days of sales recorded in last 28 days
    df_output["14_day_forecast"] = df_output["14_day_forecast_prelim"]
    df_output.loc[
        df_output["fc_14_in_28_days"] == False, "14_day_forecast"
    ] = np.nan
    return df_output


def create_eval_metric_columns(
    df_input: pd.DataFrame, sales_adj_col: str
) -> pd.DataFrame:
    df_output = df_input.copy()
    df_output["eval_threshold_5_prc"] = df_output["7_day_forecast"] * 0.05
    df_output["7_day_forecast_thr"] = df_output["7_day_forecast"]
    df_output.loc[
        (df_output[sales_adj_col] < df_output["eval_threshold_5_prc"]),
        "7_day_forecast_thr",
    ] = 0.01
    df_output["mape_7_day_forecast"] = abs(
        (df_output[sales_adj_col] - df_output["7_day_forecast"])
        / df_output[sales_adj_col]
    )
    df_output["mape_14_day_forecast"] = abs(
        (df_output[sales_adj_col] - df_output["14_day_forecast"])
        / df_output[sales_adj_col]
    )
    df_output["real_diff"] = df_output[sales_adj_col] - df_output["7_day_forecast"]
    df_output["abs_diff"] = abs(df_output[sales_adj_col] - df_output["7_day_forecast"])
    df_output["real_perc_diff"] = (
        100 * (df_output["real_diff"]) / df_output[sales_adj_col]
    )
    df_output["abs_perc_diff"] = (
        100 * abs(df_output["abs_diff"]) / df_output[sales_adj_col]
    )

    return df_output


def etl_pipeline(
    df_sales: pd.DataFrame,
    df_meta: pd.DataFrame,
    df_hols: pd.DataFrame,
    df_calendar: pd.DataFrame,
    unique_id_sales: str,
    unique_id_meta: str,
    sales_col: str,
) -> pd.DataFrame:
    df_comb = merge_sales_meta(df_sales, unique_id_sales, df_meta, unique_id_meta)
    unique_ids = get_unique_business_ids(df_comb, unique_id_sales)
    sales_adj_col = f"{sales_col}_adj"
    dfs_ls = []
    for id_ in unique_ids:
        df_comb_ = df_comb[df_comb[unique_id_sales] == id_].copy()
        df_comb_ = join_calendar_to_sales_history(df_comb_, df_calendar)
        df_comb_ = join_hols_to_sales_history_calendar(df_comb_, df_hols)
        df_comb_ = create_lead_up_columns(df_comb_)
        df_comb_ = zero_negative_sales(df_comb_, sales_col)
        df_comb_ = get_last_5_weeks_sales_per_day(df_comb_, sales_adj_col)
        df_comb_ = flag_14_out_of_28_days_sales_history(df_comb_, sales_adj_col)
        df_comb_ = create_7_day_forecast_columns(df_comb_)
        df_comb_ = create_14_day_forecast_columns(df_comb_)
        df_comb_ = create_eval_metric_columns(df_comb_, sales_adj_col)
        dfs_ls.append(df_comb_)
    df_output = pd.concat(dfs_ls, axis=0)
    return df_output
