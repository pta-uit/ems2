import argparse
from flask import Flask, request, jsonify
import pandas as pd
from s3fs.core import S3FileSystem
from datetime import datetime, timedelta

app = Flask(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Start the Solar Prediction API server')
    parser.add_argument('--predictions', type=str, required=True, help='S3 path to the predictions CSV file')
    return parser.parse_args()

def load_predictions_from_s3(s3_path):
    s3 = S3FileSystem()
    with s3.open(s3_path, 'r') as f:
        df = pd.read_csv(f)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

predictions_df = None

@app.route('/predict', methods=['GET'])
def predict():
    start_datetime = request.args.get('datetime')
    n_hours = int(request.args.get('n_hours', 24))

    if not start_datetime:
        return jsonify({"error": "datetime parameter is required"}), 400

    try:
        start_datetime = pd.to_datetime(start_datetime)
    except ValueError:
        return jsonify({"error": "Invalid datetime format"}), 400

    end_datetime = start_datetime + timedelta(hours=n_hours)

    mask = (predictions_df['timestamp'] >= start_datetime) & (predictions_df['timestamp'] < end_datetime)
    filtered_predictions = predictions_df[mask]

    if filtered_predictions.empty:
        return jsonify({"error": "No predictions available for the specified time range"}), 404

    predictions_list = filtered_predictions.to_dict(orient='records')
    return jsonify(predictions_list)

if __name__ == '__main__':
    args = parse_args()
    predictions_df = load_predictions_from_s3(args.predictions)
    app.run(debug=True, host='0.0.0.0', port=5000)
