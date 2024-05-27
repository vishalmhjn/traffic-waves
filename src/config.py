from datetime import datetime, timedelta
import os
from pathlib import Path

# API URL
URL = (
    "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/"
    "comptages-routiers-permanents/records"
)

# Previous day's input data i.e., to make predictions for today, we use yesterday's data
INFERENCE_DATA_DATE = (datetime.today() - timedelta(1)).strftime("%Y-%m-%d")

# Path handling

# Define the base paths
data_folder = Path("../data")

# Define the specific paths using the base paths
file_raw_input = data_folder / "raw_data" / f"raw_data_{INFERENCE_DATA_DATE}.csv"
file_train_input = data_folder / "historical_data" / "paris_trunk_june_july.csv"
file_static_attributes = data_folder / "processed_data" / "link_static_attributes.csv"
file_historical_trends = data_folder / "processed_data" / "link_historical_trends.csv"
file_processed_input = (
    data_folder / "processed_data" / f"inference_data_{INFERENCE_DATA_DATE}.csv"
)

# column names
list_column_order = [
    "time_idx",
    "day",
    "hour",
    "maxspeed",
    "length",
    "lanes",
    "paris_id",
    "q",
]

# Network detector IDs used to query the data
LINKS = [
    "5169",
    "5173",
    "5175",
    "5176",
    "5181",
    "5182",
    "5183",
    "5184",
    "5185",
    "5186",
    "5187",
    "5205",
    "5207",
    "5208",
    "5214",
    "5215",
    "5232",
    "5233",
    "5236",
    "5264",
    "5266",
    "5273",
    "5275",
    "5277",
    "5279",
    "5282",
    "5289",
    "5298",
    "5299",
    "5301",
    "5302",
    "5308",
    "5310",
    "5312",
    "5314",
    "5315",
    "5317",
    "5319",
    "5322",
    "5324",
    "5325",
    "5326",
    "5327",
    "5328",
    "5329",
    "5330",
    "5331",
    "5332",
    "5333",
    "5334",
    "5335",
    "5336",
    "5337",
    "5338",
    "5343",
    "5346",
    "5353",
    "5354",
    "5355",
    "5360",
    "5361",
    "5362",
    "5363",
    "5364",
    "5365",
    "5366",
    "5367",
    "5370",
    "5375",
    "5376",
    "5377",
    "5378",
    "5379",
    "5382",
    "5384",
    "5385",
    "5386",
    "5387",
    "5388",
    "5389",
    "5390",
    "5391",
    "5392",
    "5393",
    "5394",
    "5395",
    "5404",
    "5405",
    "5406",
    "5407",
    "5408",
    "5409",
    "5410",
    "5411",
    "5412",
    "5413",
    "5414",
    "5415",
    "5419",
    "5421",
    "5425",
    "5431",
    "5432",
    "5433",
    "5434",
    "5435",
    "5436",
    "5437",
    "5438",
    "5439",
    "5441",
    "5446",
    "5450",
    "5455",
    "5456",
]
