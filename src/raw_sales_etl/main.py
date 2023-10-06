from datetime import date
from pathlib import Path
import logging
import os

import pandas as pd
from holidays_api.holidays import create_combined_holidays_df
from sales_data_etl.etl import create_calendar_df
from sales_data_etl.etl import etl_pipeline
from sales_data_etl.etl import read_meta_from_disk
from sales_data_etl.etl import read_sales_from_disk

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

base_dir = "/opt/ml/processing"


def main() -> None:
    sd = date(2021, 1, 1)
    ed = date(2023, 12, 31)
    df_cal = create_calendar_df(sd, ed)
    # Update to path in S3
    path_hourly_sales_data = os.path.join(base_dir, "input", "hourly_sales_per_branch")
    filepath_hourly_sales_data = os.path.join(
        path_hourly_sales_data, os.listdir(path_hourly_sales_data)[0]
    )

    logger.info(
        f"Loading forecasts data from path '{filepath_hourly_sales_data}' into DataFrame..."
    )

    df = pd.read_csv(filepath_hourly_sales_data)
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
    path_meta_data = os.path.join(base_dir, "input", "hourly_sales_per_branch")
    filepath_meta_data = os.path.join(path_meta_data, os.listdir(path_meta_data)[0])

    logger.info(f"Loading meta data from path '{filepath_meta_data}' into DataFrame...")
    df_m = read_meta_from_disk(Path(filepath_meta_data))
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
        holiday_scaling_factor_contribution=1,
    )
    # df_output.to_csv("AWS/S3/Location/holiday_scaling_factors_2021_2023.csv")
    output_filename = f"holiday_scaling_factors_2021_2023_{date.today()}.csv"
    logger.info(f"Storing forecasts to {output_filename}...")

    df_output.to_csv(f"{base_dir}/output/{output_filename}")


if __name__ == "__main__":
    main()
