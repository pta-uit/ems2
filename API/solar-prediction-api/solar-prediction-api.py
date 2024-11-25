import argparse
from flask import Flask, request, jsonify
import pandas as pd
from s3fs.core import S3FileSystem
from datetime import datetime, timedelta
import os

app = Flask(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Start the Solar Prediction API server')
    parser.add_argument('--predictions', type=str, required=True, help='S3 directory containing prediction CSV files')
    return parser.parse_args()

def load_predictions_from_s3(s3_dir):
    s3 = S3FileSystem()
    predictions = {}
    
    # Load LSTNET predictions
    lstnet_path = os.path.join(s3_dir, 'lstnet_predictions.csv')
    attention_path = os.path.join(s3_dir, 'attention_predictions.csv')
    
    with s3.open(lstnet_path, 'r') as f:
        lstnet_df = pd.read_csv(f)
        lstnet_df['timestamp'] = pd.to_datetime(lstnet_df['timestamp'])
        predictions['lstnet'] = lstnet_df
        
    with s3.open(attention_path, 'r') as f:
        attention_df = pd.read_csv(f)
        attention_df['timestamp'] = pd.to_datetime(attention_df['timestamp'])
        predictions['attention'] = attention_df
        
    return predictions

predictions_dict = None

@app.route('/predict', methods=['GET'])
def predict():
    start_datetime = request.args.get('datetime')
    n_hours = int(request.args.get('n_hours', 24))
    model = request.args.get('model', 'lstnet')  # Default to lstnet if not specified

    if not start_datetime:
        return jsonify({"error": "datetime parameter is required"}), 400
        
    if model not in ['lstnet', 'attention']:
        return jsonify({"error": "Invalid model. Choose 'lstnet' or 'attention'"}), 400

    try:
        start_datetime = pd.to_datetime(start_datetime)
    except ValueError:
        return jsonify({"error": "Invalid datetime format"}), 400

    end_datetime = start_datetime + timedelta(hours=n_hours)
    
    # Get predictions for the specified model
    predictions_df = predictions_dict[model]
    
    mask = (predictions_df['timestamp'] >= start_datetime) & (predictions_df['timestamp'] < end_datetime)
    filtered_predictions = predictions_df[mask]

    if filtered_predictions.empty:
        return jsonify({"error": "No predictions available for the specified time range"}), 404

    predictions_list = filtered_predictions.to_dict(orient='records')
    return jsonify(predictions_list)

if __name__ == '__main__':
    args = parse_args()
    predictions_dict = load_predictions_from_s3(args.predictions_dir)
    app.run(debug=True, host='0.0.0.0', port=5000)
