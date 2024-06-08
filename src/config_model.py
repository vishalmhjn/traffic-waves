TRAINING_PARAMS = {
    "metric": "smape",
    "training": True,
    "model_output_dir": "model_output",
    "seed": 46,
    "test_proportion": 0.15,
    "validation_proportion": 0.15,
    "patience": 25,
    "train_episodes": 1000,
    "batch_size": 512,
}

FORECASTING_PARAMS = {
    "lb": 24,
    "ph": 24,
}

## month and day are considered as static for the forecasting horizon, short-term forecasting
FEATURE_SETTINGS = {
    "dyn_to_static": ["day", "hour"],
    "include_occupancy": False,
    "dynamic_categorical_features": ["day", "hour"],
    "static_categorical_features": [],
    "dynamic_continous_features": [
        # "speed_kph_mean",
        # "speed_kph_stddev",
        "q",
    ],
    "static_continous_features": [
        "maxspeed",
        "lanes",
        "length",
    ],
    "other_columns": ["time_idx", "paris_id"],
    "occupancy_column": ["k"],
    "target_as_autoregressive_feature": ["q"],
    "target_column": ["qt"],
}

# Using the CONFIG dictionary
metric = TRAINING_PARAMS["metric"]
training = TRAINING_PARAMS["training"]
train_episodes = TRAINING_PARAMS["train_episodes"]

dynamic_continous_features = FEATURE_SETTINGS["dynamic_continous_features"]
dynamic_categorical_features = FEATURE_SETTINGS["dynamic_categorical_features"]
static_continous_features = FEATURE_SETTINGS["static_continous_features"]
static_categorical_features = FEATURE_SETTINGS["static_categorical_features"]

continous_features = [*static_continous_features, *dynamic_continous_features]
categorical_features = [*static_categorical_features, *dynamic_categorical_features]

dyn_to_static = FEATURE_SETTINGS["dyn_to_static"]

occupancy_column = FEATURE_SETTINGS["occupancy_column"]
other_columns = FEATURE_SETTINGS["other_columns"]
target_as_autoregressive_feature = FEATURE_SETTINGS["target_as_autoregressive_feature"]
target_column = FEATURE_SETTINGS["target_column"]

if not FEATURE_SETTINGS["include_occupancy"]:
    pass
else:
    dynamic_continous_features.append(*occupancy_column)

if FEATURE_SETTINGS["include_occupancy"]:
    dynamic_features = [
        # TODO: ordering as first: categorical and second: continous in the list
        # is important for preprocess data fucntion
        *dynamic_categorical_features,
        *dynamic_continous_features,
        *occupancy_column,
        *target_column,
    ]
else:
    dynamic_features = [
        # TODO: ordering as first: categorical and second: continous in the list
        # is important for preprocess data fucntion
        *dynamic_categorical_features,
        *dynamic_continous_features,
        *target_column,
    ]
static_features = [*static_categorical_features, *static_continous_features]
