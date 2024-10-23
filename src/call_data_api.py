"""
Module: call_data_api

This module provides functions for data collection from API.
"""

import os
from dataclasses import dataclass
import requests
import pandas as pd

from utils import setup_logging
from config_data import URL, LINKS

logging = setup_logging(file_name="call_data_api.log")


@dataclass
class ParisAPIHandler:
    """call data api"""

    url: str
    params: dict
    save_path: str
    link_id: int

    def __repr__(self):
        return (
            f"ParisAPIHandler(url='{self.url}', params={self.params}, "
            f"save_path='{self.save_path}', link_id={self.link_id})"
        )

    def call_open_api(self):
        """
        Make a GET request to the specified URL with parameters
        and save the response data to a CSV file.
        """
        response = requests.get(self.url, params=self.params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            df = pd.json_normalize(data["results"])
            df.to_csv(f"{self.save_path}/raw_data_{self.link_id}.csv", index=False)
        else:
            logging.info("Error: %s", response.status_code)


@dataclass
class DataMerger:
    """process collected data"""

    path: str
    list_links: list
    read_path: str
    date_formatted: str

    def merge_data(self):
        """
        Merge data from multiple CSV files into a single CSV file.
        """
        full_data_list = []
        for i in self.list_links:
            df = pd.read_csv(f"{self.read_path}/raw_data_{i}.csv")
            df["t_1h"] = pd.to_datetime(df["t_1h"])
            assert (
                str(df["t_1h"].dt.date.min()) == self.date_formatted
            ), f"Data for previous day is not available via API yet. For \
                other days, manually set the offsets in API query."

            full_data_list.append(df)
        full_data = pd.concat(full_data_list, axis=0)
        full_data.to_csv(self.path, index=False)

    def clean_data(self):
        """
        Delete raw data from multiple CSV files.
        """
        for i in self.list_links:
            file_path = f"{self.read_path}/raw_data_{i}.csv"
            # Check if the file exists before deleting
            if os.path.exists(file_path):
                # Delete the file
                os.remove(file_path)
                logging.info("File %s deleted successfully.", file_path)
            else:
                logging.info("File %s does not exist.", file_path)


def data_collector(
    raw_data_folder,
    input_date,
    limit=24,
    offset=0,
    timezone="Europe/Berlin",
):
    """Wrapper fucntion to collect and save the data"""
    for link in LINKS:

        # Define the query parameters
        params = {
            "select": "iu_ac,t_1h,q",
            "where": f"iu_ac in ('{link}')",
            "limit": limit,
            "order_by": "t_1h DESC",
            "timezone": timezone,
            "offset": offset,
            # for date filtering: t_1h >= '2024-06-11 01:00:00' AND\
            #  t_1h <= '2024-06-12 00:00:00' AND iu_ac in ('5169')
        }
        # Example of using the dataclasses
        api_handler = ParisAPIHandler(
            url=URL, params=params, save_path=raw_data_folder, link_id=link
        )

        # Call the methods of the dataclasses
        api_handler.call_open_api()

    data_merger = DataMerger(
        path=f"{raw_data_folder}/raw_data_{input_date}.csv",
        list_links=LINKS,
        read_path=raw_data_folder,
        date_formatted=input_date,
    )
    data_merger.merge_data()
    data_merger.clean_data()
    logging.info(f"Data for {input_date} downloaded successfully.")
