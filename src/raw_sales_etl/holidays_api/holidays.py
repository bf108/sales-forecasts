from datetime import date
from datetime import datetime

import pandas as pd
from holidays_config import CALENDAR_META  # pylint: disable=import-error
from holidays_config import CALENDAR_URL  # pylint: disable=import-error


def create_calendar_df(start_dt: date, end_dt: date) -> pd.DataFrame:
    df_calendar = pd.DataFrame(pd.date_range(start_dt, end_dt, freq="d")).rename(
        columns={0: "date"}
    )
    df_calendar.loc[:, "dummy"] = "calendar"
    df_calendar.set_index("date", inplace=True)
    return df_calendar


def convert_ms_unix_timestamp_to_datatime(unix_timestamp: str) -> datetime:
    ts_ = int(unix_timestamp)
    return datetime.strptime(
        datetime.utcfromtimestamp(ts_ / 1e3).strftime("%Y-%m-%d"), "%Y-%m-%d"
    )


def create_url(country: str, year: int) -> str:
    if country.lower() in CALENDAR_META:
        meta = CALENDAR_META[country.lower()]
        return f"{CALENDAR_URL}/{country}/{year}?hol={meta}"
    return f"{CALENDAR_URL}/{country}/{year}"
