from datetime import date
from pathlib import Path
from typing import Union
import calendar

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
    df_output = df_calendar.join(df_sales, how="left")
    cols = [c for c in df_output.columns if c not in ["y", "operational_flag"]]
    df_output[cols] = df_output[cols].ffill().bfill()
    df_output.drop(columns=["dummy"], inplace=True)
    df_output["operational_flag"] = df_output["operational_flag"].ffill()
    assert (
        df_output.shape[0] == df_calendar.shape[0]
    ), f"Missing rows from join: Calendar: {df_calendar.shape[0]} vs Output {df_output.shape[0]}"

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
    df_output["date_column_bhw"] = df_output.index
    df_output["dow_int"] = df_output["date_column_bhw"].dt.dayofweek
    df_output.loc[df_output["flag_holiday"].isna(), "flag_holiday"] = False
    df_output["flag_lead_up_holiday"] = df_output["flag_holiday"].shift(periods=-1)
    df_output["lead_up_holiday_name"] = (
        df_output["holiday_name"].shift(periods=-1).fillna("")
    )
    df_output["three_day_shift"] = df_output["flag_holiday"].shift(periods=-3)
    df_output["three_day_shift_name"] = (
        df_output["holiday_name"].shift(periods=-3).fillna("")
    )
    df_output["two_day_shift"] = df_output["flag_holiday"].shift(periods=-2)
    df_output["two_day_shift_name"] = (
        df_output["holiday_name"].shift(periods=-2).fillna("")
    )
    df_output["bank_holiday_weekend_name_thurs_prior"] = df_output.apply(
        lambda x: f"thur_prior_to_{x['lead_up_holiday_name']}"
        if (
            (x["dow_int"] == 3)
            and (x["flag_lead_up_holiday"] == True)
            and (not x["lead_up_holiday_name"].startswith("Valentine"))
        )
        else None,
        axis=1,
    )
    df_output["bank_holiday_weekend_name_fri_prior"] = df_output.apply(
        lambda x: f"fri_prior_to_{x['three_day_shift_name']}"
        if (
            (x["dow_int"] == 4)
            and (x["three_day_shift"] == True)
            and (not x["three_day_shift_name"].startswith("Valentine"))
        )
        else None,
        axis=1,
    )
    df_output["bank_holiday_weekend_name_sat_prior"] = df_output.apply(
        lambda x: f"sat_prior_to_{x['two_day_shift_name']}"
        if (
            (x["dow_int"] == 5)
            and (x["two_day_shift"] == True)
            and (not x["two_day_shift_name"].startswith("Valentine"))
        )
        else None,
        axis=1,
    )
    df_output["bank_holiday_weekend_name_sun_prior"] = df_output.apply(
        lambda x: f"sun_prior_to_{x['lead_up_holiday_name']}"
        if (
            (x["dow_int"] == 6)
            and (x["flag_lead_up_holiday"] == True)
            and (not x["lead_up_holiday_name"].startswith("Valentine"))
        )
        else None,
        axis=1,
    )
    df_output["holiday_name_v1"] = (
        df_output["holiday_name"]
        .combine_first(df_output["bank_holiday_weekend_name_thurs_prior"])
        .combine_first(df_output["bank_holiday_weekend_name_fri_prior"])
        .combine_first(df_output["bank_holiday_weekend_name_sat_prior"])
        .combine_first(df_output["bank_holiday_weekend_name_sun_prior"])
    )

    return df_output


def get_last_7_weeks_sales_per_day(
    df_input: pd.DataFrame, sales_col: str
) -> pd.DataFrame:
    df_output = df_input.copy()
    df_output["sales_7_days_prior"] = df_output[sales_col].shift(7)
    df_output["sales_14_days_prior"] = df_output[sales_col].shift(14)
    df_output["sales_21_days_prior"] = df_output[sales_col].shift(21)
    df_output["sales_28_days_prior"] = df_output[sales_col].shift(28)
    df_output["sales_35_days_prior"] = df_output[sales_col].shift(35)
    df_output["sales_42_days_prior"] = df_output[sales_col].shift(42)
    df_output["sales_49_days_prior"] = df_output[sales_col].shift(49)
    return df_output


def flag_14_out_of_28_days_sales_history(
    df_input: pd.DataFrame, sales_col: str
) -> pd.DataFrame:
    df_output = df_input.copy()
    df_output["fc_14_in_28_days"] = (
        ~df_output[sales_col].rolling(window=28, min_periods=14).sum().isna()
    )
    return df_output


def x_day_forecast(df_input: pd.DataFrame, x: int):
    df_output = df_input.copy()
    x_cp = x
    counter = 0
    input_cols = []
    while x >= 7:
        cols = [f"sales_{(i*7)+x}_days_prior" for i in range(4)]
        if counter > 0:
            input_cols.append(f"{x_cp}_day_forecast_input_{counter}")
            cols = input_cols + cols[counter:]
        input_col = f"{x_cp}_day_forecast_input_{counter+1}"
        df_output[input_col] = df_output[cols].mean(axis=1)
        x -= 7
        counter += 1
    df_output[f"{x_cp}_day_forecast"] = df_output[
        f"{x_cp}_day_forecast_input_{counter}"
    ]
    df_output.loc[
        df_output["fc_14_in_28_days"] == False, f"{x_cp}_day_forecast"
    ] = np.nan
    return df_output


def create_eval_metric_columns(
    df_input: pd.DataFrame, sales_adj_col: str, sales_adj_prc_th: float
) -> pd.DataFrame:
    df_output = df_input.copy()
    for day in [7, 14, 21, 28]:
        n_day_forecast = f"{day}_day_forecast"
        n_day_eval_th = f"{day}_day_eval_threshold"
        n_day_forecast_th = f"{day}_day_forecast_threshold"
        n_day_ape = f"{day}_day_forecast_abs_error"
        n_day_re = f"{day}_day_forecast_real_error"
        n_day_a_diff = f"{day}_day_abs_diff"
        n_day_r_diff = f"{day}_day_real_diff"

        df_output[n_day_eval_th] = df_output[n_day_forecast] * sales_adj_prc_th
        df_output[n_day_forecast_th] = df_output[n_day_forecast]
        df_output.loc[
            (df_output[sales_adj_col] < df_output[n_day_eval_th]), n_day_forecast_th
        ] = np.nan
        df_output[n_day_ape] = abs(
            (df_output[sales_adj_col] - df_output[n_day_forecast_th])
            / df_output[sales_adj_col]
        )
        df_output[n_day_re] = (
            df_output[sales_adj_col] - df_output[n_day_forecast_th]
        ) / df_output[sales_adj_col]
        df_output[n_day_a_diff] = abs(
            df_output[sales_adj_col] - df_output[n_day_forecast_th]
        )
        df_output[n_day_r_diff] = (
            df_output[sales_adj_col] - df_output[n_day_forecast_th]
        )

    return df_output


def add_year_month_day_columns(df_input: pd.DataFrame) -> pd.DataFrame:
    df_output = df_input.copy()
    dow_mapping = {i: d for i, d in enumerate(calendar.day_abbr)}
    month_mapping = {i: d for i, d in enumerate(calendar.month_abbr)}
    df_output["year"] = df_output.index.year
    df_output["dow"] = [dow_mapping[v] for v in df_output.index.day_of_week.values]
    # df_output['week_id'] = df_output.index.week
    df_output["month_id"] = df_output.index.month
    df_output["month"] = [month_mapping[v] for v in df_output.index.month.values]
    return df_output


def operating_day_in_365_days(df_input: pd.DataFrame) -> pd.DataFrame:
    df_output = df_input.copy()
    df_output["operating_days_365_lookforward"] = (
        (~df_output["y"].isna()).rolling("365D", min_periods=1).sum().shift(-365).values
    )
    return df_output


def join_venue_operational_stats(df_input: pd.DataFrame) -> pd.DataFrame:
    df_output = df_input.copy()
    # Join max operating days, date of start and finish of plateau and days at top to each unqiue_id
    res_max_lookforward = []
    for i, _id in enumerate(df_output["unique_id"].unique()):
        tmp_df = df_output[(df_output["unique_id"] == _id)].copy()
        tmp_df_index = tmp_df.index
        max_days = tmp_df["operating_days_365_lookforward"].max()
        first_max_index = tmp_df_index[
            tmp_df["operating_days_365_lookforward"].argmax()
        ]
        last_max_index = tmp_df_index[::-1][
            tmp_df["operating_days_365_lookforward"][::-1].argmax()
        ]
        operating_2020 = tmp_df[tmp_df["date_column"] == "2020-01-01"][
            "operating_days_365_lookforward"
        ].max()
        operating_2021 = tmp_df[tmp_df["date_column"] == "2021-01-01"][
            "operating_days_365_lookforward"
        ].max()
        operating_2022 = tmp_df[tmp_df["date_column"] == "2022-01-01"][
            "operating_days_365_lookforward"
        ].max()
        res_max_lookforward.append(
            {
                "unique_id": _id,
                "max_operating_days": max_days,
                "date_first_max_op_days": first_max_index,
                "date_last_max_op_days": last_max_index,
                "op_days_2020": operating_2020,
                "op_days_2021": operating_2021,
                "op_days_2022": operating_2022,
            }
        )
    df_max_lookforward = pd.DataFrame(res_max_lookforward)
    df_max_lookforward["plateau_length_days"] = (
        df_max_lookforward["date_last_max_op_days"]
        - df_max_lookforward["date_first_max_op_days"]
    ).dt.days
    df_output_v1 = pd.merge(
        df_output,
        df_max_lookforward,
        left_on="unique_id",
        right_on="unique_id",
        how="left",
    )
    df_output_v1.index = df_output.index
    return df_output_v1


def country_level_holiday_factors(df_input: pd.DataFrame, year: int) -> pd.DataFrame:
    df_output = df_input.copy()
    df_hol_scaling = (
        df_input[(~df_input["holiday_name_v1"].isna()) & (df_input["year"] == year)]
        .groupby(by=["holiday_name_v1", "country"])
        .agg({"7_day_forecast_real_error": ["median"]})
        .droplevel(0, 1)[["median"]]
        .reset_index()
        .dropna()
        .rename(columns={"median": f"holiday_scaling_factor_country_median_{year}"})
    )
    df_output_v1 = df_output.merge(
        df_hol_scaling,
        left_on=["country", "holiday_name_v1"],
        right_on=["country", "holiday_name_v1"],
        how="left",
    )
    df_output_v1.index = df_output.index
    return df_output_v1


def brand_level_holiday_factors(df_input: pd.DataFrame, year: int) -> pd.DataFrame:
    df_output = df_input.copy()
    # Take mean performance across all branches in year
    df_hol_scaling = (
        df_output[(~df_output["holiday_name_v1"].isna()) & (df_output["year"] == year)]
        .groupby(by=["brandname", "holiday_name_v1"])
        .agg({"7_day_forecast_real_error": ["mean", "median"]})
        .droplevel(0, 1)[["mean", "median"]]
        .reset_index()
        .dropna()
        .rename(
            columns={
                "mean": f"holiday_scaling_factor_brand_mean_{year}",
                "median": f"holiday_scaling_factor_brand_median_{year}",
            }
        )
    )
    df_output_v1 = df_output.merge(
        df_hol_scaling,
        left_on=["brandname", "holiday_name_v1"],
        right_on=["brandname", "holiday_name_v1"],
        how="left",
    )
    df_output_v1.index = df_output.index
    return df_output_v1


def branch_level_holiday_factors(
    df_input: pd.DataFrame, years: list[int]
) -> pd.DataFrame:
    df_output = df_input.copy()
    df_hol_scaling = (
        df_output[
            (~df_output["holiday_name_v1"].isna()) & (df_output["year"].between(*years))
        ]
        .groupby(by=["brandname", "branchname", "holiday_name_v1"])
        .agg({"7_day_forecast_real_error": ["mean", "median"]})
        .droplevel(0, 1)[["mean", "median"]]
        .reset_index()
        .dropna()
        .rename(
            columns={
                "mean": "holiday_scaling_factor_branch_mean",
                "median": "holiday_scaling_factor_branch_median",
            }
        )
    )
    df_output_v1 = df_output.merge(
        df_hol_scaling,
        left_on=["brandname", "holiday_name_v1"],
        right_on=["brandname", "holiday_name_v1"],
        how="left",
    )
    df_output_v1.index = df_output.index
    return df_output_v1


def etl_pipeline(
    df_sales: pd.DataFrame,
    df_meta: pd.DataFrame,
    df_hols: pd.DataFrame,
    df_calendar: pd.DataFrame,
    unique_id_sales: str,
    unique_id_meta: str,
    sales_col: str,
    sales_adj_prc_th: float = None,
) -> pd.DataFrame:
    df_comb = merge_sales_meta(df_sales, unique_id_sales, df_meta, unique_id_meta)
    unique_ids = get_unique_business_ids(df_comb, unique_id_sales)
    sales_adj_col = f"{sales_col}_adj"
    dfs_ls = []
    # Default to 5% of average in last 4 weeks
    if not sales_adj_prc_th:
        sales_adj_prc_th = 0.05
    for id_ in unique_ids:
        df_comb_ = df_comb[df_comb[unique_id_sales] == id_].copy()
        df_comb_ = join_calendar_to_sales_history(df_comb_, df_calendar)
        df_comb_ = join_hols_to_sales_history_calendar(df_comb_, df_hols)
        df_comb_ = create_lead_up_columns(df_comb_)
        df_comb_ = zero_negative_sales(df_comb_, sales_col)
        df_comb_ = get_last_7_weeks_sales_per_day(df_comb_, sales_adj_col)
        df_comb_ = flag_14_out_of_28_days_sales_history(df_comb_, sales_adj_col)
        for day in [7, 14, 21, 28]:
            df_comb_ = x_day_forecast(df_comb_, day)
        df_comb_ = create_eval_metric_columns(df_comb_, sales_adj_col, sales_adj_prc_th)
        df_comb_ = operating_day_in_365_days(df_comb_)
        dfs_ls.append(df_comb_)
    df_output = pd.concat(dfs_ls, axis=0)
    df_output = add_year_month_day_columns(df_output)
    df_output["holiday_name_v1"] = (
        df_output["holiday_name_v1"]
        .str.replace(" ", "_")
        .str.replace(".", "")
        .str.lower()
    )
    df_output["date_column"] = df_output.index
    df_output = join_venue_operational_stats(df_output)
    df_output = country_level_holiday_factors(df_output, 2022)
    df_output = brand_level_holiday_factors(df_output, 2022)
    df_output = branch_level_holiday_factors(df_output, [2021, 2022])
    return df_output
