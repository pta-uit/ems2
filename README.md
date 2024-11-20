##################### this line is for testing
## Solar Generation Forecasting with LSTNet

### Mainly referenced paper
Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks.(https://arxiv.org/abs/1703.07015)

### Dataset
NeurIPS 2022: CityLearn Challenge ([starter-kit-team-Together](https://gitlab.aicrowd.com/aicrowd/challenges/citylearn-challenge/citylearn-2022-starter-kit-team-together/-/tree/master/data/citylearn_challenge_2022_phase_1?ref_type=heads)); Building_1.csv & weather.csv

### Model flow diagram
![Architecture](https://i.imgur.com/UIQqYqp.png)

### Requirements
```
boto3==1.35.28
botocore==1.35.23
hyperopt==0.2.7
numpy==2.1.1
pandas==2.2.3
s3fs==2024.9.0
scikit_learn==1.5.2
statsmodels==0.14.3
torch==2.4.1+cu121
```

### Usage
#### 1. Clone this repo:
```
git clone https://github.com/pta-uit/solar-lstnet.git
```
#### 2. Install requirements:
```
pip install -r requirements.txt
```
#### 3. Set up S3 environment:
```
import os
import boto3
import s3fs

# Set up AWS credentials
os.environ['AWS_ACCESS_KEY_ID'] = 'YOURS3ACCESSKEYID'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'YOURS3SECRETACCESSKEY'

# Initialize S3 client
s3 = boto3.client('s3')

# Initialize S3 filesystem
fs = s3fs.S3FileSystem(anon=False)

print("S3 setup complete.")
```
#### 4. Data validation:
```
python data_validation.py --weather_data s3://path/weather.csv --building_data s3://path/Building_1.csv --output_report s3://path/validation_report.json
```
#### 5. Data preprocessing:
```
python data_preprocessing.py --year 2022 --weather_data s3://path/weather_cleaned.csv --building_data s3://path/Building_1_cleaned.csv --solar --output s3://path/
```
- `--year`: start year of the dataset, set to current year if not specified
- `--solar`: to tell the script if we're preprocessing data for training (include Solar Generation) or preprocessing data for predicting (not include Solar Generation)
#### 6. Hyperparameter tuning:
```
python hyperparameter_tuning.py --preprocessed_data s3://path/preprocessed_data.pkl --max_evals 10 --save s3://path/
```
- `max_evals`: Maximum number of evaluations for hyperparameter tuning
#### 7. Training:
```
python main.py --preprocessed_data s3://path/preprocessed_data.pkl --best_params s3://path/best_params.json --save s3://path/
```
- `best_params`: path to the best hyperparameters JSON file, train using default hyperparameters if not specicified
#### 8. Prepare input data for prediction:
```
python input_preparation.py --historical_data s3://path/preprocessed_data.pkl --weather_forecast_data s3://path/preprocessed_weather_only.pkl --datetime "2022-06-01 00:00:00" --h 168 --f 24 --output s3://path/prepared_data.pkl
```
- We supposed that `h` hours before `datetime` in the preprocessed_data (which preprocessed with `--solar`) is historical data, and `f` hours after `datetime` in the weather_forecast_data (which preprocessed without `--solar`) are weather forecast data
- Then the script will concatenate them to produce the input_data
#### 9. Prediction
```
python predict.py --model_path s3://path/model.pt --input_data s3://path/prepared_data.pkl --output s3://path/predictions.csv --strategy weighted_average --lambda_param 0.1
```
- `strategy`: Strategy for processing predictions (single, average, most_recent, weighted_average),
- `lambda_param`: Lambda parameter for weighted average strategy
- The script will process predictions using weighted_average by default.

#### Plot first sample:
![Forecast-plot](https://i.imgur.com/HHENEZd.png)