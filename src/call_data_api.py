"""
Module: call_data_api

This module provides functions for data collection from API.
"""

import os
from dataclasses import dataclass
import requests
from tqdm import tqdm

import pandas as pd

from utils import setup_logging
from config import URL, LINKS, INFERENCE_DATA_DATE, config_folder


temp_path = os.path.join(config_folder, "../data/raw_data")

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

    def merge_data(self):
        """
        Merge data from multiple CSV files into a single CSV file.
        """
        full_data_list = []
        for i in self.list_links:
            df = pd.read_csv(f"{self.read_path}/raw_data_{i}.csv")
            df["t_1h"] = pd.to_datetime(df["t_1h"])

            if str(df["t_1h"].dt.date.min()) == INFERENCE_DATA_DATE:
                full_data_list.append(df)
            else:
                logging.info("Data for %s detector is not available", i)
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


def data_collector(limit=24, offset=0, timezone="Europe/Berlin"):
    """Wrapper fucntion to collect and save the data"""
    for link in tqdm(LINKS):

        # Define the query parameters
        params = {
            "select": "iu_ac,t_1h,q",
            "where": f"iu_ac in ('{link}')",
            "limit": limit,
            "order_by": "t_1h DESC",
            "timezone": timezone,
            "offset": offset,
        }
        # Example of using the dataclasses
        api_handler = ParisAPIHandler(
            url=URL, params=params, save_path=temp_path, link_id=link
        )

        # Call the methods of the dataclasses
        api_handler.call_open_api()

    data_merger = DataMerger(
        path=f"{temp_path}/raw_data_{INFERENCE_DATA_DATE}.csv",
        list_links=LINKS,
        read_path=temp_path,
    )
    data_merger.merge_data()
    data_merger.clean_data()


if __name__ == "__main__":
    data_collector()
