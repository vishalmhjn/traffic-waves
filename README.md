# Traffic Waves

Traffic-Waves is a voluntary project focused on daily traffic predictions in Paris, utilizing data from [Open Data Paris](https://opendata.paris.fr/explore/dataset/comptages-routiers-permanents/information)

## Overview

The project leverages ML and DL techniques to analyze historical traffic data and make predictions for daily traffic patterns in Paris. This aids in providing insights for commuters and city planners alike.

## Download and Install

To install the requirements, run the following command in the parent:

```bash
git clone https://github.com/vishalmhjn/traffic-waves.git
cd traffic-waves
make install
```

## Usage

Run the data collection, processing and machine learing pipeline:
```bash
make run
```

Run the visualization app:
```bash
python app.py run
```

## Pipeline components

The above command runs the following components:

**Data**:
- **[Data collection](src/call_data_api.py)**: Call the Open data Paris API and save the data in batches.
- **[Data processing](src/process_data.py)**: Merge the data and apply preprocessing steps to prepare data for batch predictions.

**Machine learning**:
- **[Model training](src/train.py)**: Import and train the ML model on the historical data.
- **[Predictions](src/predict.py)**: Get the one-day ahead predictions using the trained model and batch data.

**Visualization**:
- **[Dashboard](src/app.py)**: Start a flask app to display the input data and predictions for all the links.

## License
**to be updated**