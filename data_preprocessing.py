import argparse
import pickle
import json
from utils.data_util import DataUtil
import os
import time
import pandas as pd
from s3fs.core import S3FileSystem

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess data for LSTNet Solar Generation Forecasting')
    parser.add_argument('--weather_data', type=str, required=True, help='Path to the weather CSV file')
    parser.add_argument('--building_data', type=str, required=True, help='Path to the building CSV file')
    parser.add_argument('--solar', action='store_true', help='Include solar generation data')
    parser.add_argument('--year', type=int, default=None, help='Year of the data')
    parser.add_argument('--window', type=int, default=168, help='Window size (default: 168)')
    parser.add_argument('--horizon', type=int, default=24, help='Forecasting horizon (default: 24)')
    parser.add_argument('--output', type=str, required=True, help='S3 path to save the preprocessed data')
    return parser.parse_args()

def save_to_s3(s3_path, data):
    s3 = S3FileSystem()
    with s3.open(s3_path, 'wb') as f:
        f.write(pickle.dumps(data))

def save_json_to_s3(s3_path, data):
    s3 = S3FileSystem()
    with s3.open(s3_path, 'w') as f:
        json.dump(data, f, indent=2)

def generate_report(df, X, y, features, target, window, horizon, data_type):
    return {
        "data_type": data_type,
        "start_datetime": str(df.index.min()),
        "end_datetime": str(df.index.max()),
        "total_samples": len(df),
        "features": features,
        "target": target,
        "window_size": window,
        "horizon": horizon,
        "X_shape": X.shape,
        "y_shape": y.shape if y is not None else None,
        "data_frequency": pd.infer_freq(df.index),
        "missing_values": df.isnull().sum().to_dict(),
        "feature_statistics": df[features].describe().to_dict() if features else None,
        "target_statistics": df[target].describe().to_dict() if target else None
    }

def main():
    start_time = time.time()
    args = parse_args()

    print("Starting data preprocessing...")
    data_util = DataUtil(args.weather_data, args.building_data, args.year, args.solar)
    df = data_util.load_and_preprocess_data()
    df = data_util.perform_feature_engineering()
    print("Data preprocessing and feature engineering completed.")

    features = ['Outdoor Drybulb Temperature [C]', 'Relative Humidity [%]',
                'Diffuse Solar Radiation [W/m2]', 'Direct Solar Radiation [W/m2]',
                'hour_sin', 'hour_cos', 'day_of_year_sin', 'day_of_year_cos', 'month',
                'trend', 'seasonal', 'residual']
    target = 'Solar Generation [W/kW]' if args.solar else None

    X, y = data_util.prepare_sequences(args.window, args.horizon, features, target)
    print(f"Sequences prepared. X shape: {X.shape}, y shape: {y.shape if y is not None else None}")

    preprocessed_data = {
        'X': X,
        'y': y,
        'features': features,
        'target': target,
        'scaler': data_util.scaler,
        'window': args.window,
        'horizon': args.horizon,
        'year': data_util.year
    }
    
    filename = "preprocessed_data.pkl" if args.solar else "preprocessed_weather_only.pkl"
    data_type = "Weather and Solar Generation Data" if args.solar else "Weather Data Only"
    
    print(f"Saving preprocessed data to S3...")
    output_path = os.path.join(args.output, filename)
    try:
        save_to_s3(output_path, preprocessed_data)
        print(f"Successfully saved preprocessed data to {output_path}")
    except Exception as e:
        print(f"Error saving preprocessed data to S3: {e}")

    report = generate_report(df, X, y, features, target, args.window, args.horizon, data_type)
    report['year'] = data_util.year
    report_filename = os.path.splitext(filename)[0] + '.json'
    report_path = os.path.join(args.output, report_filename)
    try:
        save_json_to_s3(report_path, report)
        print(f"Successfully saved report to {report_path}")
    except Exception as e:
        print(f"Error saving report to S3: {e}")

    data_util.verify_preprocessing()
    print("Preprocessing verification completed.")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Preprocessing task completed in {total_time:.2f} seconds")

if __name__ == "__main__":
    main()