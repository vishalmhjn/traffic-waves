from flask import Flask, render_template
import pandas as pd
import argparse

from config_data import (
    PATH_PREDICTIONS,
    input_date_formatted,
    prediction_date_formatted,
    file_processed_input,
    file_sample_variance,
)
from frontend import DashboardData

pd.options.mode.chained_assignment = None

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model",
    help="type of machine learning model",
    choices=["knn", "xgboost"],
    default="knn",
)
args = parser.parse_args()

dashboard_object = DashboardData(
    path_o_t_1=file_processed_input,
    path_pt_1=(PATH_PREDICTIONS / f"{args.model}_{input_date_formatted}.csv"),
    path_pt=(PATH_PREDICTIONS / f"{args.model}_{prediction_date_formatted}.csv"),
    path_variance=file_sample_variance,
)


dashboard_object.read_data()

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/data.json")
def data():
    data_asset = dashboard_object.processing_pipeline()

    return DashboardData.write_to_json("../frontend/data.json", data_asset)


@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


if __name__ == "__main__":
    app.run(debug=True)
