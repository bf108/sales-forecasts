from datetime import date
from pathlib import Path

import pandas as pd
from holidays_api.holidays import create_combined_holidays_df
from sales_data_etl.etl import create_calendar_df
from sales_data_etl.etl import etl_pipeline
from sales_data_etl.etl import read_meta_from_disk
from sales_data_etl.etl import read_sales_from_disk


def main() -> None:
    sd = date(2021, 1, 1)
    ed = date(2024, 9, 28)
    df_cal = create_calendar_df(sd, ed)
    # Update to path in S3
    df = pd.read_csv("../raw_sales_etl/data/hourly_sales_per_branch.csv")
    df["ds"] = pd.to_datetime(df["ds"])
    df["ds"] = df["ds"].dt.date
    df_s = (
        df.groupby(by=["unique_id", "ds"])["y"]
        .sum()
        .reset_index()
        .sort_values(by=["ds"], ascending=True)
        .reset_index(drop=True)
    )
    # Update to path in S3
    df_m = read_meta_from_disk(Path("../raw_sales_etl/data/branches.csv"))
    df_hols = create_combined_holidays_df(
        countries=["uk", "ireland", "malta", "spain"], years=[2021, 2022, 2023]
    )
    df_output = etl_pipeline(
        df_sales=df_s,
        df_meta=df_m,
        df_hols=df_hols,
        df_calendar=df_cal,
        unique_id_sales="unique_id",
        unique_id_meta="_id",
        sales_col="y",
        sales_adj_prc_th=0.05,
        holiday_scaling_factor_contribution=0.5,
    )
    df_output.to_csv("AWS/S3/Location/holiday_scaling_factors_2021_2023.csv")


if __name__ == "__main__":
    main()
