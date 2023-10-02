from datetime import date
from pathlib import Path
import pandas as pd
from sales_data_etl.etl import (
    read_meta_from_disk,
    read_sales_from_disk,
    create_calendar_df,
    etl_pipeline,
)
from holidays_api.holidays import create_combined_holidays_df


def main() -> None:
    sd = date(2021, 1, 1)
    ed = date(2024, 9, 28)
    df_cal = create_calendar_df(sd, ed)
    #Update to path in AWS
    df = pd.read_csv('../raw_sales_etl/data/hourly_sales_per_branch.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    df['ds'] = df['ds'].dt.date
    df_s = (df.groupby(by=['unique_id','ds'])['y'].sum().reset_index().sort_values(by=['ds'], ascending=True)
            .reset_index(drop=True))
    #Update to path in AWS
    # df_s = read_sales_from_disk(
    #     Path("../raw_sales_etl/data/daily_sales_data_all_branches.csv")
    # )
    #Update to path in AWS
    df_m = read_meta_from_disk(Path("../raw_sales_etl/data/branches.csv"))
    df_hols = create_combined_holidays_df(countries=["uk", "ireland"], years=[2021, 2022, 2023])
    df_output = etl_pipeline(
        df_sales=df_s,
        df_meta=df_m,
        df_hols=df_hols,
        df_calendar=df_cal,
        unique_id_sales="unique_id",
        unique_id_meta="_id",
        sales_col="y",
        sales_adj_prc_th=0.05
    )
    print(df_output.shape)

if __name__ == "__main__":
    main()
