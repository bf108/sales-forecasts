from ast import literal_eval
from datetime import date
from datetime import datetime

import pandas as pd
import requests  # type: ignore
from bs4 import BeautifulSoup as bs

CALENDAR_URL = "https://www.timeanddate.com/holidays"
HOLIDAY_GRANULARITY = {
    1: "Official holidays",
    9: "Official holidays and non-working days",
    25: "Holidays and some observances",
    4194329: "Holidays (incl. some local) and observances",
    4194331: "Holidays (incl. all local) and observances",
    313: "Holidays and many observances",
    13759295: "All holidays/observances/religious events",
}
CALENDAR_META = {"uk": 4194329, "ireland": 281, "malta": 25}
IMPORTANT_DAYS = [
    "Boxing Day",
    "Christmas Day",
    "Christmas Eve",
    "Day off for New Years Day",
    "Early August Bank Holiday",
    "Early May Bank Holiday",
    "Easter Monday",
    "Easter Sunday",
    "Fathers Day",
    "Good Friday",
    "June Bank Holiday",
    "Late August Bank Holiday",
    "Mothers Day",
    "New Years Day",
    "New Years Day observed",
    "New Years Eve",
    "October Bank Holiday",
    "Public Holiday",
    "Spring Bank Holiday",
    "St. Brigids Day",
    "St. Patricks Day",
    "Substitute Bank Holiday for Boxing Day",
    "Substitute Bank Holiday for Christmas Day",
    "Valentines Day",
]


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
    table_headings = [f"holiday_{t}" for t in _table_headings]
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
    df_holiday_table.loc[:, "flag_holiday"] = True
    df_holiday_table.loc[:, "country"] = country
    df_holiday_table.drop(columns=["holiday_dow"], inplace=True)
    df_holiday_table.set_index("date", inplace=True)
    return df_holiday_table[~df_holiday_table.index.duplicated()]


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
    df_combined = pd.concat(holidays_dfs, axis=0)
    df_combined = clean_up_holidays_df(df_combined)
    df_combined = add_valentines_ireland(df_combined, years)
    return df_combined


def add_valentines_ireland(df_input: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    df_output = df_input.copy()
    ireland_val = []
    for y in years:
        tmp_dict = {
            "date": date(y, 2, 14),
            "holiday_name": "Valentines Day",
            "country": "ireland",
            "flag_holiday": True,
        }
        ireland_val.append(tmp_dict)

    df_tmp = pd.DataFrame(ireland_val)
    df_tmp["date"] = pd.to_datetime(df_tmp["date"])
    df_tmp.set_index("date", inplace=True)
    df_output = pd.concat([df_output, df_tmp])
    return df_output


def clean_up_holidays_df(df_input: pd.DataFrame) -> pd.DataFrame:
    df_output = df_input.copy()
    scottish_bank_holidays = df_output[
        (df_output["holiday_name"] == "Summer Bank Holiday")
        & (df_output["holiday_details"] == "Scotland")
    ].index
    df_output.loc[scottish_bank_holidays, "holiday_name"] = "Early August Bank Holiday"
    # Process holiday data
    df_output["holiday_name"] = df_output["holiday_name"].str.replace(
        "'|’", "", regex=True
    )
    df_output["holiday_name"] = df_output["holiday_name"].str.replace(
        "Summer Bank Holiday", "Late August Bank Holiday", regex=True
    )
    df_output["holiday_name"] = df_output["holiday_name"].str.replace(
        "^August Bank Holiday$", "Early August Bank Holiday", regex=True
    )
    df_output["holiday_name"] = df_output["holiday_name"].str.replace(
        "^May Day$", "Early May Bank Holiday", regex=True
    )
    df_output["holiday_name"] = df_output["holiday_name"].str.replace(
        "^St. Stephens Day$", "Boxing Day", regex=True
    )
    df_output["holiday_name"] = df_output["holiday_name"].str.replace(
        " / VE Day", "", regex=True
    )
    df_output["holiday_name"] = df_output["holiday_name"].str.replace(
        "^Easter$", "Easter Sunday", regex=True
    )
    df_output = df_output[(df_output["holiday_name"].isin(IMPORTANT_DAYS))].drop(
        columns=["holiday_details", "holiday_type"]
    )
    return df_output


def join_holidays_df_to_calendar_df(
    df_calendar: pd.DataFrame, df_holidays: pd.DataFrame
) -> pd.DataFrame:
    df_new = df_calendar.join(df_holidays, how="left").drop(columns=["dummy"])
    # Check correct number of days after join
    assert (
        df_new.shape[0] == df_calendar.shape[0]
    ), f"df_new shape {df_new.shape[0]} df_calendar shape {df_calendar.shape[0]}"

    # Check no duplicate indices (dates)
    assert df_new.index.duplicated().any() is False

    hol_flag_cols = [col for col in df_new if col.startswith("flag_")]
    df_new.loc[:, hol_flag_cols] = df_new[hol_flag_cols].fillna(False)

    for col in hol_flag_cols:
        country = col.split("_")[-1]
        df_new[f"flag_lead_up_{country}"] = df_new[col].shift(periods=-1)
        df_new[f"lead_up_holiday_name_{country}"] = df_new[
            f"holiday_name_{country}"
        ].shift(periods=-1)
    return df_new
