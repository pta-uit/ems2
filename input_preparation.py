import argparse
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from s3fs.core import S3FileSystem
import pickle

def get_s3(s3_path):
    s3 = S3FileSystem()
    return pickle.load(s3.open(s3_path, 'rb'))

def load_s3(s3_path, data):
    s3 = S3FileSystem()
    with s3.open(s3_path, 'wb') as f:
        pickle.dump(data, f)

def load_json_report(s3_path):
    s3 = S3FileSystem()
    with s3.open(s3_path, 'r') as f:
        return json.load(f)

def prepare_input_data(historical_data_path, weather_forecast_data_path, target_date, h, f):
    historical_data = get_s3(historical_data_path)
    weather_forecast_data = get_s3(weather_forecast_data_path)

    historical_report = load_json_report(historical_data_path.replace('.pkl', '.json'))
    weather_forecast_report = load_json_report(weather_forecast_data_path.replace('.pkl', '.json'))

    target_datetime = pd.to_datetime(target_date)

    historical_features = historical_report['features']
    forecast_features = weather_forecast_report['features']

    historical_start_time = pd.to_datetime(historical_report['start_datetime'])
    forecast_start_time = pd.to_datetime(weather_forecast_report['start_datetime'])
    
    historical_hours = int((target_datetime - historical_start_time).total_seconds() / 3600)
    forecast_hours = int((target_datetime - forecast_start_time).total_seconds() / 3600)
    
    historical_start_index = max(0, historical_hours - h)
    historical_end_index = historical_hours
    forecast_start_index = max(0, forecast_hours)
    forecast_end_index = forecast_start_index + f

    historical_slice = historical_data['X'][historical_start_index:historical_end_index]
    forecast_slice = weather_forecast_data['X'][forecast_start_index:forecast_end_index]

    common_features = list(set(historical_features) & set(forecast_features))
    historical_feature_indices = [historical_features.index(feat) for feat in common_features]
    forecast_feature_indices = [forecast_features.index(feat) for feat in common_features]

    historical_slice = historical_slice[:, :, historical_feature_indices]
    forecast_slice = forecast_slice[:, :, forecast_feature_indices]

    combined_data = np.concatenate([historical_slice, forecast_slice], axis=0)

    if 'y' in historical_data and historical_report['target'] is not None:
        historical_target = historical_data['y'][historical_start_index:historical_end_index]
        solar_generation = np.zeros((combined_data.shape[0], combined_data.shape[1], 1))
        for i in range(len(historical_target)):
            day_index = i % 7
            solar_generation[i, day_index*24:(day_index+1)*24, 0] = historical_target[i]
        combined_data = np.concatenate([combined_data, solar_generation], axis=2)
        common_features.append('Solar Generation [W/kW]')

    target_length = h + f
    current_length = combined_data.shape[0]
    if current_length < target_length:
        pad_length = target_length - current_length
        padding = np.zeros((pad_length, combined_data.shape[1], combined_data.shape[2]))
        combined_data = np.concatenate([combined_data, padding], axis=0)
    elif current_length > target_length:
        combined_data = combined_data[:target_length]

    time_steps = 168
    num_samples = combined_data.shape[0] // time_steps
    combined_data = combined_data[:num_samples * time_steps].reshape(num_samples, time_steps, -1)

    prepared_data = {
        'X': combined_data,
        'features': common_features,
        'start_datetime': (historical_start_time + timedelta(hours=historical_start_index)).isoformat(),
        'target': 'Solar Generation [W/kW]' if 'y' in historical_data else None,
        'h': h,
        'f': f
    }
    
    return prepared_data

def main():
    parser = argparse.ArgumentParser(description="Prepare input data for prediction.")
    parser.add_argument("--historical_data", required=True, help="S3 path to historical data pickle file")
    parser.add_argument("--weather_forecast_data", required=True, help="S3 path to weather forecast data pickle file")
    parser.add_argument("--datetime", required=True, help="Target datetime (format: YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--h", type=int, default=168, help="Number of historical hours to include")
    parser.add_argument("--f", type=int, default=24, help="Number of forecast hours to include")
    parser.add_argument("--output", default='s3://trambk/solar-energy/preprocessed_data/prepared_data.pkl', help="S3 path for output pickle file")

    args = parser.parse_args()

    try:
        prepared_data = prepare_input_data(args.historical_data, args.weather_forecast_data, 
                                           args.datetime, args.h, args.f)

        load_s3(args.output, prepared_data)
        print(f"Prepared data saved to {args.output}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()