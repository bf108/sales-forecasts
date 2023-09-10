from ast import literal_eval
from datetime import date
from datetime import datetime

import pandas as pd
import requests  # type: ignore
from bs4 import BeautifulSoup as bs
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


def get_holiday_granularity_options_for_country(
    country: str, year: int
) -> dict[int, str]:
    url = create_url(country, year)
    resp = requests.get(url, timeout=60)
    soup = bs(resp.content, "html.parser")
    options = soup.find("select", id="hol").find_all("option")
    return {int(opt["value"]): opt.text for opt in options[:-1]}


def get_table_headings(soup: bs) -> list[str]:
    table_headings = [
        h.text.lower()
        for h in soup.find(id="holidays-table").find("thead").find("tr").find_all("th")
    ]
    # Replace empty field with day of week (dow)
    table_headings[1] = "dow"
    return table_headings


def create_country_holidays_df(country: str, year: int) -> pd.DataFrame:
    url = create_url(country, year)
    resp = requests.get(url, timeout=60)
    soup = bs(resp.content, "html.parser")
    _table_headings = get_table_headings(soup)
    table_headings = [f"holiday_{t}_{country}" for t in _table_headings]
    res = []
    # Find table of holidays in html
    # iterate through rows of table and extract content
    for row in (
        soup.find(id="holidays-table").find("tbody").find_all("tr", class_="showrow")
    ):
        row_dict = {"date": convert_ms_unix_timestamp_to_datatime(row["data-date"])}
        for value, heading in zip(row.find_all("td"), table_headings[1:]):
            row_dict[heading] = value.text
        res.append(row_dict)
    df_holiday_table = pd.DataFrame(res)
    df_holiday_table.loc[:, f"flag_holiday_{country}"] = True
    df_holiday_table.drop(columns=[f"holiday_dow_{country}"], inplace=True)
    df_holiday_table.set_index("date", inplace=True)
    return df_holiday_table


def join_list_of_df(dfs: list[pd.DataFrame], how: str = "outer") -> pd.DataFrame:
    if len(dfs) == 0:
        raise ValueError("Empty list of DataFrames Passed")
    if len(dfs) == 1:
        return dfs[0]
    return literal_eval(
        "dfs[0]"
        + "".join([f".join(dfs[{i+1}], how='{how}')" for i, _ in enumerate(dfs[1:])])
    )


def create_combined_holidays_df(countries: list[str], years: list[int]) -> pd.DataFrame:
    holidays_dfs = []
    for country in countries:
        tmp_dfs = []
        for year in years:
            tmp_dfs.append(create_country_holidays_df(country, year))
        tmp_df = pd.concat(tmp_dfs, axis=0)
        holidays_dfs.append(tmp_df)
    df_combined = join_list_of_df(holidays_dfs)
    return df_combined[~df_combined.index.duplicated(keep="first")]
