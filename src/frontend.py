from abc import ABC, abstractmethod

import simplejson
import json
from flask import jsonify
import math
import numpy as np
import pandas as pd

from config_data import prediction_date, input_date


class PlotFormatting(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def read_data():
        pass


class DashboardData(PlotFormatting):
    _prediction_date = prediction_date
    _input_date = input_date

    def __init__(self, path_pt, path_pt_1, path_o_t_1, path_variance=None) -> None:
        self.path_predictions_t = path_pt
        self.path_predictions_t_1 = path_pt_1
        self.path_observed_t_1 = path_o_t_1
        self.path_variance = path_variance
        super().__init__()
        self.set_dates()

    @staticmethod
    def create_date_strings():
        current_date = DashboardData._prediction_date.strftime("%d-%m-%Y")
        previous_date = DashboardData._input_date.strftime("%d-%m-%Y")
        return previous_date, current_date

    @classmethod
    def set_dates(cls):
        cls.previous_date, cls.current_date = cls.create_date_strings()

    def read_data(self):
        self.df_observed_t_1 = pd.read_csv(self.path_observed_t_1)
        self.df_predictions_t_1 = pd.read_csv(self.path_predictions_t_1)
        self.df_predictions_t = pd.read_csv(self.path_predictions_t)
        if self.path_variance != None:
            self.df_variance = pd.read_csv(self.path_variance)

    @staticmethod
    def filter_df(df, i):
        return df[df.paris_id == i]

    def get_time_and_target(self, filtered_df, col_name, date):
        filtered_df.loc[:, "new_time_idx"] = filtered_df["time_idx"].apply(
            lambda idx: f"{date}:{idx:02d}"
        )
        time_idx = filtered_df["new_time_idx"].tolist()
        target_value = filtered_df[col_name].tolist()
        return time_idx, target_value

    @staticmethod
    def get_variance(df):
        var_t_1 = var_t = df["q"].tolist()
        var_t.extend(var_t_1)
        return var_t

    @staticmethod
    def create_confidence_intervals(predictions, variance):
        ci_upper = list(np.array(predictions) + np.array(variance))
        ci_lower = list(np.array(predictions) - np.array(variance))

        ci_upper = [int(np.maximum(x, 0)) if not math.isnan(x) else x for x in ci_upper]
        ci_lower = [int(np.maximum(x, 0)) if not math.isnan(x) else x for x in ci_lower]
        return ci_lower, ci_upper

    def processing_pipeline(self):
        data = []
        for paris_id in self.df_observed_t_1["paris_id"].unique():

            try:
                temp_o = DashboardData.filter_df(self.df_observed_t_1, paris_id)
                temp_p_t = DashboardData.filter_df(self.df_predictions_t, paris_id)
                temp_p_t_1 = DashboardData.filter_df(self.df_predictions_t_1, paris_id)
                temp_var = DashboardData.filter_df(self.df_variance, paris_id)

                time_idx_o, q_o = self.get_time_and_target(
                    temp_o, "q_real", DashboardData.previous_date
                )
                time_idx_p_t_1, q_p_t_1 = self.get_time_and_target(
                    temp_p_t_1, "preds", DashboardData.previous_date
                )
                time_idx_p_t, q_p_t = self.get_time_and_target(
                    temp_p_t, "preds", DashboardData.current_date
                )

                time_idx_p_t_1.extend(time_idx_p_t)
                q_p_t_1.extend(q_p_t)

                q_o = DashboardData.array_to_list(q_o)
                q_p_t_1 = DashboardData.array_to_list(q_p_t_1)

                if self.path_variance != None:
                    var_preditctions = DashboardData.get_variance(temp_var)
                    ci_lower, ci_upper = DashboardData.create_confidence_intervals(
                        q_p_t_1, var_preditctions
                    )

                data.append(
                    {
                        "paris_id": int(paris_id),
                        "real_time_idx": time_idx_o,
                        "real_q": q_o,
                        "predictions_time_idx": time_idx_p_t_1,
                        "predictions_preds": q_p_t_1,
                        "lower_bound": ci_lower,
                        "upper_bound": ci_upper,
                    }
                )
            except ValueError:
                pass
        return data

    @staticmethod
    def array_to_list(X):
        return [int(x) if not math.isnan(x) else x for x in X]

    @staticmethod
    def write_to_json(path, data):
        with open(path, "w") as json_file:
            normalized_str = simplejson.dumps(data, ignore_nan=True)
            data = simplejson.loads(normalized_str)
            json.dump(data, json_file)  # , ignore_nan=True)
        return jsonify(data)
