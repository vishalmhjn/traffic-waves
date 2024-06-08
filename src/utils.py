import logging
from itertools import repeat

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def setup_logging(file_name="logfile.log"):
    """Set up logging"""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=file_name,
        filemode="w",
    )
    return logging


def predicitons_to_df(ph, z_test, y_test_hat):
    df_test = pd.DataFrame({"paris_id": [x for x in z_test for _ in repeat(None, ph)]})

    df_test["time_idx"] = np.tile(np.arange(ph), len(z_test))
    df_test["preds"] = np.ravel(y_test_hat)
    df_test["preds"] = df_test["preds"].astype(int)
    return df_test


def format_dates(offset_days=0):
    base_date = datetime.today() - timedelta(offset_days)
    return base_date, base_date.strftime("%Y-%m-%d")


def func_path_data(raw_folder, date_fmt, prefix_str):
    return raw_folder / f"{prefix_str}_{date_fmt}.csv"
