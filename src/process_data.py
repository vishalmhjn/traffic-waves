"""Data processing recipe generated from the Jupyter Notebook
"""

import os
import pandas as pd

from utils import setup_logging
from config_data import LIST_COLUMN_ORDER

logging = setup_logging("logfile_datapreprocess.log")


def create_static_attributes(file_static, file_train):
    """Create static attribute file"""

    if os.path.exists(file_static):
        logging.info("File %s exists.", file_static)
    else:
        logging.info("File %s does not exist.", file_static)
        _df = pd.read_csv(file_train)
        temp = _df.drop(
            columns=[
                "time_idx",
                "month",
                "day",
                "hour",
                "speed_kph_mean",
                "speed_kph_stddev",
                "q",
            ]
        )
        temp.drop_duplicates(subset=["paris_id"], inplace=True)
        temp.reset_index(drop=True, inplace=True)
        temp.to_csv(file_static, index=False)
        logging.info("File %s created.", file_static)


def compute_historical_trends(file_historical, file_train):
    """compute historical trends"""

    if os.path.exists(file_historical):
        logging.info("File %s exists.", file_historical)
    else:
        logging.info("File %s does not exist.", file_historical)
        _df = pd.read_csv(file_train)
        df_trends = _df.groupby(by=["paris_id", "day", "hour"]).mean()
        df_trends["q"] = df_trends["q"].astype(int)
        df_trends.reset_index(inplace=True)
        df_trends.to_csv(file_historical, index=False)
        logging.info("File %s created.", file_historical)


def merge_files(file_static, file_raw, file_historcal, list_columns):
    """Merge the two tables based on paris_id, day, and hour"""

    temp = pd.read_csv(file_static)
    df_real = pd.read_csv(file_raw)
    df_trends = pd.read_csv(file_historcal)
    _df_merged = pd.merge(
        left=temp, right=df_real, left_on="paris_id", right_on="iu_ac"
    )

    _df_merged["t_1h"] = pd.to_datetime(_df_merged["t_1h"])
    _df_merged["time_idx"] = (
        _df_merged["t_1h"] - _df_merged["t_1h"].min()
    ).dt.total_seconds() / 3600

    # Convert to integer
    _df_merged["time_idx"] = _df_merged["time_idx"].astype(int)

    _df_merged["day"] = _df_merged["t_1h"].dt.dayofweek
    _df_merged["hour"] = _df_merged["t_1h"].dt.hour

    _df_merged = _df_merged[list_columns]

    _df_sorted = _df_merged.sort_values(by=["time_idx", "paris_id"])
    _df_sorted.reset_index(inplace=True, drop=True)

    _merged_df = _df_sorted.merge(
        df_trends,
        on=["paris_id", "day", "hour"],
        how="left",
        suffixes=("_sorted", "_trends"),
    )
    return _merged_df


def fill_missing_values(_df):
    """Fill missing values in columns of df_sorted from corresponding columns in df_merged"""

    columns_to_fill = ["q"]
    for column in columns_to_fill:
        _df[column + "_real"] = _df[column + "_sorted"]
        _df[column + "_sorted"] = _df[column + "_sorted"].fillna(
            _df[column + "_trends"]
        )

    return _df


def data_processor(
    file_historical_trends,
    file_static_attributes,
    file_train_input,
    file_raw_input,
    file_processed_input,
):
    """wrapper function"""
    create_static_attributes(file_static_attributes, file_train_input)
    compute_historical_trends(file_historical_trends, file_train_input)
    df = merge_files(
        file_static_attributes,
        file_raw_input,
        file_historical_trends,
        LIST_COLUMN_ORDER,
    )

    merged_df = fill_missing_values(df)

    # Drop columns from df_merged that were used for filling missing values
    for col in [
        "time_idx_trends",
        "month",
        "maxspeed_trends",
        "length_trends",
        "lanes_trends",
        "speed_kph_mean",
        "speed_kph_stddev",
        "q_trends",
    ]:
        merged_df.drop(columns=[col], axis=1, inplace=True)

    merged_df.rename(
        columns={col: col.replace("_sorted", "") for col in merged_df.columns},
        inplace=True,
    )

    merged_df.to_csv(file_processed_input, index=False)
    logging.info("Inference input file %s created.", file_processed_input)
