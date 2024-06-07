from flask import Flask, render_template
import pandas as pd

from config_data import input_date_formatted, prediction_date_formatted
from frontend import DashboardData

pd.options.mode.chained_assignment = None

dashboard_object = DashboardData(
    path_o_t_1=f"../data/processed_data/inference_data_{input_date_formatted}.csv",
    path_pt_1=f"../predictions/knn_{input_date_formatted}.csv",
    path_pt=f"../predictions/knn_{prediction_date_formatted}.csv",
    path_variance=f"../data/variance/df_var_2023.csv",
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
