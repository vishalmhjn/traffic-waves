from main import (
    raw_data_folder,
    processed_data_folder,
)
from main import format_dates, func_path_data


previous_date, previous_date_formatted = format_dates(day_delta=1)
current_date, current_date_formatted = format_dates()
file_raw_input = func_path_data(raw_data_folder, previous_date_formatted, "raw_data")
file_processed_input = func_path_data(
    processed_data_folder, previous_date_formatted, "inference_data"
)
