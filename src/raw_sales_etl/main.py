from datetime import date
from pathlib import Path
from sales_data_etl.etl import read_meta_from_disk, read_sales_from_disk, create_calendar_df, etl_pipeline
from holidays_api.holidays import create_combined_holidays_df


def main() -> None:
    sd = date(2021,1,1)
    ed = date(2024,7,15)
    df_cal = create_calendar_df(sd, ed)
    df_s = read_sales_from_disk(Path('../raw_sales_etl/data/daily_sales_data_all_branches.csv'))
    df_m = read_meta_from_disk(Path('../raw_sales_etl/data/branches.csv'))
    df_hols = create_combined_holidays_df(['uk','ireland'],[2021,2022,2023])
    df_output = etl_pipeline(df_s, df_m, df_hols, df_cal, 'unique_id', '_id', 'y', sales_adj_prc_th=0.15)
    return df_output


if __name__ == "__main__":
    main()
