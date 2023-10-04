from datetime import datetime

import country_converter as coco
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
CALENDAR_META = {"uk": 4194329, "ireland": 4194329, "malta": 4194329}


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
    # soup.find(id="holidays-table").find("tbody").find_all("tr", class_="showrow")
    for row in soup.find(id="holidays-table").find("tbody").find_all("tr"):
        try:
            row_dict = {"date": convert_ms_unix_timestamp_to_datatime(row["data-date"])}
            for value, heading in zip(row.find_all("td"), table_headings[1:]):
                row_dict[heading] = value.text
            res.append(row_dict)
        except:
            pass
    df_holiday_table = pd.DataFrame(res)
    df_holiday_table.set_index("date", inplace=True)
    df_holiday_table = normalize_text(df_holiday_table)
    df_holiday_table = form_unique_holiday_name(df_holiday_table)
    df_holiday_table.loc[:, "flag_holiday"] = True
    df_holiday_table.loc[:, "country"] = country
    df_holiday_table.drop(
        columns=["holiday_dow", "holiday_type", "holiday_details"],
        inplace=True,
        errors="ignore",
    )
    return df_holiday_table[~df_holiday_table.index.duplicated()]


def form_unique_holiday_name(df_input: pd.DataFrame) -> pd.DataFrame:
    df_output = df_input.copy()
    for col in df_output.columns:
        if col not in ["date", "holiday_name", "holiday_dow"]:
            df_output["holiday_name"] = df_output["holiday_name"].str.cat(
                df_output[col], sep="_"
            )
    df_output["holiday_name"] = df_output["holiday_name"].str.rstrip("_")
    return df_output


def normalize_text(df_input: pd.DataFrame) -> pd.DataFrame:
    df_output = df_input.copy()
    for col in df_output.columns:
        if col != "date":
            df_output[col] = (
                df_output[col]
                .str.replace(" ", "_")
                .str.replace(".", "")
                .str.replace("'", "")
                .str.replace("`", "")
                .str.replace(",", "")
                .str.replace("’", "")
                .str.lower()
                .str.strip()
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


def create_lead_up_days(df_input: pd.DataFrame) -> pd.DataFrame:
    df_output = df_input.copy()
    df_output.reset_index(inplace=True)
    df_output["dow_int"] = df_output["date"].dt.dayofweek

    df_monday = df_output[df_output["dow_int"] == 0].reset_index(drop=True).copy()
    day_dict_monday = {
        "sunday": {"days_offset": 1, "prefix": "sunday_prior_to_", "dow": 6},
        "saturday": {"days_offset": 2, "prefix": "saturday_prior_to_", "dow": 5},
        "friday": {"days_offset": 3, "prefix": "friday_prior_to_", "dow": 4},
    }
    dfs = []
    for _, val in day_dict_monday.items():
        df_tmp = df_monday.copy()
        df_tmp["date"] = df_tmp["date"] - pd.Timedelta(days=val["days_offset"])
        df_tmp["dow_int"] = val["dow"]
        df_tmp["holiday_name"] = val["prefix"] + df_tmp["holiday_name"]
        dfs.append(df_tmp)
    df_lead_up_monday = pd.concat(dfs)

    df_friday = df_output[df_output["dow_int"] == 4].reset_index(drop=True).copy()
    day_dict_friday = {
        "thursday": {"days_offset": 1, "prefix": "thursday_prior_to_", "dow": 3},
    }
    dfs = []
    for _, val in day_dict_friday.items():
        df_tmp = df_friday.copy()
        df_tmp["date"] = df_tmp["date"] - pd.Timedelta(days=val["days_offset"])
        df_tmp["dow_int"] = val["dow"]
        df_tmp["holiday_name"] = val["prefix"] + df_tmp["holiday_name"]
        dfs.append(df_tmp)
    df_lead_up_friday = pd.concat(dfs)

    df_output_v1 = pd.concat([df_output, df_lead_up_monday, df_lead_up_friday])
    df_output_v1 = (
        df_output_v1.drop_duplicates(subset=["date", "country"])
        .sort_values(by=["date"])
        .reset_index(drop=True)
    )
    return df_output_v1


def convert_country_name_to_iso3166_alpha_2(df_input: pd.DataFrame) -> pd.DataFrame:
    converter = coco.CountryConverter()
    df_output = df_input.copy()
    df_output["cc"] = df_output["country"].apply(
        lambda x: converter.convert(x, to="iso2")
    )
    return df_output


def create_combined_holidays_df(
    *, countries: list[str], years: list[int]
) -> pd.DataFrame:
    holidays_dfs = []
    for country in countries:
        tmp_dfs = []
        for year in years:
            tmp_dfs.append(create_country_holidays_df(country, year))
        tmp_df = pd.concat(tmp_dfs, axis=0)
        holidays_dfs.append(tmp_df)
    df_combined = pd.concat(holidays_dfs, axis=0)
    df_combined = create_lead_up_days(df_combined)
    df_combined = convert_country_name_to_iso3166_alpha_2(df_combined)
    df_combined.set_index(["date", "cc"], inplace=True)
    df_combined.drop(columns=["dow_int", "country"], inplace=True)
    return df_combined
