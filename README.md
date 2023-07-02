# Energy Modeling For Buildings
This script performs energy modeling for buildings using combinations of relevant machine learning algorithms. It takes in cleaned CSV files of building energy data and produces trained models for energy usage prediction.

## Docker Usage
Build the Docker image using the provided Dockerfile in the project directory:
```shell
docker build -t energy-modeling .
docker run -v /dev/shm:/dev/shm --shm-size=2gb -d -p 80:8080 energy-modeling
```

## Setup & Dependencies
- Conda
    - https://conda.io/projects/conda/en/latest/user-guide/install/index.html
- Tensorflow (Steps Down Below)
    - https://www.tensorflow.org/install/pip#linux

```bash
conda create -n tf-gpu --yes python==3.9
conda activate tf-gpu
conda update --yes --all
conda install --yes -c conda-forge prophet
conda install --yes numpy pandas matplotlib flask

pip3 install --upgrade pip
python3 -m pip install --upgrade setuptools
pip3 install -r requirements.txt
```

Linux/WSL2:
```bash
conda install --yes -c conda-forge cudatoolkit=11.8.0
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# Verify install:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

MacOS:
```bash
# There is currently no official GPU support for MacOS.
python3 -m pip install tensorflow
# Verify install:
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

## Usage
1. Place preprocessed CSV files in a local data folder (e.g., `./clean_data_extended/`).
2. Verify `config.py` to ensure it has the correct configurations for your project.
3. Run the script with the following command:

- To run Flask App
```shell
python3 -m flask run --host=0.0.0.0 --port=8080
```

- To run inside terminal
```shell
python3 main.py [--preprocess] [--train] [--save] [--predict] [...]
```

### Flags
The script supports the following flags:

1. --preprocess: Performs the preprocessing step on the preprocessed CSV files in the data folder. The preprocessed data will be stored in temporary files for later use in the training step.
2. --train: Performs the training step using the preprocessed data from the previous step. It uses the temporary files generated during the preprocessing step.

Examples:
- To run only the preprocessing step:
```shell
python3 main.py --preprocess
```
- To run only the training step using the preprocessed data:
```shell
python3 main.py --train
```
- To run both preprocessing and training steps sequentially:
```shell
python3 main.py --preprocess --train
```

Heartbeat Examples:
```shell
python3 main.py --preprocess --train --save --time_step 24 --datelevel hour
python3 main.py --preprocess --train --save --time_step 30 --datelevel day
python3 main.py --preprocess --train --save --time_step 12 --datelevel month
python3 main.py --preprocess --train --save --time_step 1 --datelevel year
python3 main.py --predict --bldgname Stadium --y_column present_elec_kwh --time_step 24 --datelevel hour
```
Note: If no flags are specified, the script will display a message and exit without performing any action.

## Configuration
The script contains the following configurations:

### Settings
- data_path: Path to the directory containing the preprocessed CSV files for building energy data. The default value is f'{PATH}/clean_data_extended'.
- buildings_list: List of preprocessed CSV files for building energy data. The default value is ['Stadium_Data_Extended.csv'].
save_model_file: Boolean indicating whether to save the trained models as .pkl files. The default value is False.
- save_model_plot: Boolean indicating whether to save the model plots. The default value is False.
- min_number_of_days: Minimum number of days required for a column to be considered for training. The default value is 365.
- memory_limit: Memory limit (in KB) for the AutoSklearnRegressor. The default value is 102400.
- exclude_column: The column to exclude from training. The default value is 'present_co2_tons'.
- warnings.filterwarnings("ignore"): Ignore warnings during execution.

### Data Preprocessing
- y_columns: List of energy usage column names. The default value is ['present_elec_kwh', 'present_htwt_mmbtu', 'present_wtr_usgal', 'present_chll_tonhr', 'present_co2_tons'].
- add_features: List of additional features to include in the model. The default value is ['temp_c', 'rel_humidity_%', 'surface_pressure_hpa', 'cloud_cover_%', 'direct_radiation_w/m2', 'precipitation_mm', 'wind_speed_ground_km/h', 'wind_dir_ground_deg'].
- header: List of column names in the preprocessed CSV files. The default value is ['ts'] + y_columns + add_features.

### Training Scope
- model_types: List of model types to train. Can be 'ensembles' or 'solos'.
- preprocessing_methods: List of preprocessing methods to use. The default value is ['linear_regression', 'linear_interpolation', 'prophet', 'lstm'].
- feature_modes: List of feature selection modes. The default value is ['rfecv', 'lassocv'].

### Hyperparameters
- n_features: List of the number of features to consider. The default value is list(range(1, len(add_features))).
- n_folds: Number of folds for cross-validation. The default value is 5.
- time_steps: List of time steps to use for training. The default value is [1, 8, 12, 24].
- minutes_per_model: Maximum time in minutes for training each model. The default value is 2.
- split_rate: Train-test data split rate. The default value is 0.8.

## Input Data
The CSV files should contain the following columns:

- ts: Timestamps in YYYY-MM-DD HH:MM:SS format.
- bldgname: Building names.
- present_elec_kwh: Present electric usage in kWh.
- present_htwt_mmbtu: Present heating/cooling usage in MMBTU.
- present_wtr_usgal: Present water usage in US gallons.
- present_chll_tonhr: Present chiller usage in ton-hour.
- present_co2_tons: Present CO2 emissions in tons.
- (Optional additional features)

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
- Mean absolute percentage error (MAPE)

## Results
The trained models are saved as .pkl files in the ./models/ folder. The evaluation metrics are saved in the root project directory in results.csv