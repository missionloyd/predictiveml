# Building Energy Modeling using Prophet and AutoML
This script performs building energy modeling using Prophet and AutoML. It takes in preprocessed CSV files for building energy data and produces trained models for energy usage prediction.

## Required Libraries
- csv
- pickle
- pandas
- numpy
- matplotlib
- sklearn
- autosklearn
- fbprophet

```bash
!pip install auto-sklearn
```

If you experience any issues installing Prophet, please follow the steps to fix: [Link](https://stackoverflow.com/questions/66887159/im-trying-to-use-prophet-from-fbprophet-but-im-getting-this-excuciatingly-long).

## Usage
1. Place preprocessed CSV files in the ./clean_data/ folder.
2. Run the script with the command python Auto_ML.py.

## Configuration
The script contains the following configurations:

- buildings_list: List of preprocessed CSV files for building energy data.
- model_types: List of model types to train. Can be ensembles or solos.
- time_steps: List of time steps to use for training.
The models will be saved in the ./models/ folder according to their model type.

## Data Preprocessing
The CSV files should contain the following columns:

- ts: Timestamps in YYYY-MM-DD HH:MM:SS format.
- bldgname: Building names.
- present_elec_kwh: Present electric usage in kWh.
- present_htwt_mmbtu: Present heating/cooling usage in MMBTU.
- present_wtr_usgal: Present water usage in US gallons.
- present_chll_tonhr: Present chiller usage in ton-hour.
- present_co2_tons: Present CO2 emissions in tons.

The data is first converted into a Pandas dataframe, sorted by building name and timestamp, and grouped by building name.

The data is then split into training and testing sets, and normalized using MinMaxScaler.

## Model Training
For each building and each energy usage column, the script trains an AutoML model using AutoSklearnRegressor.

If the number of data points for an energy usage column is at least 1 year (365 days) long and the energy usage column is not present_co2_tons, missing values are filled using Prophet.

The AutoML models are trained on a sliding window of the training data using various time steps specified in time_steps.

For model_type == 'solos', only one model is trained for each energy usage column. For model_type == 'ensembles', multiple models are trained for each energy usage column and their predictions are ensembled together.

## Model Evaluation
The trained models are evaluated using various metrics, including:

- Root Mean squared error (RMSE)
- Mean absolute error (MAE)
- R-squared (R2)

## Results
The trained models are saved as .pkl files in the ./models/ folder. The evaluation metrics are saved as .txt files in the same folder.