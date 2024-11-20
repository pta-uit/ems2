import argparse
import torch
import numpy as np
import pandas as pd
from s3fs.core import S3FileSystem
from models.LSTNet import Model
from sklearn.preprocessing import MinMaxScaler
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='Make predictions using LSTNet model')
    parser.add_argument('--model_path', type=str, required=True, help='S3 path to the model file')
    parser.add_argument('--input_data', type=str, required=True, help='S3 path to the prepared input data pickle file')
    parser.add_argument('--output', type=str, required=True, help='S3 path to save the predictions')
    parser.add_argument('--strategy', type=str, default='weighted_average', choices=['single', 'average', 'most_recent', 'weighted_average'], help='Strategy for processing predictions')
    parser.add_argument('--lambda_param', type=float, default=0.1, help='Lambda parameter for weighted average strategy')
    return parser.parse_args()

def load_from_s3(s3_path):
    s3 = S3FileSystem()
    with s3.open(s3_path, 'rb') as f:
        return pickle.load(f)

def save_to_s3(data, s3_path):
    s3 = S3FileSystem()
    with s3.open(s3_path, 'w') as f:
        data.to_csv(f, index=False)

def load_model_from_s3(s3_path):
    try:
        state_dict = load_from_s3(s3_path)
        metadata_path = s3_path.replace('.pt', '_metadata.pt')
        metadata = load_from_s3(metadata_path)

        args = argparse.Namespace(**metadata['args'])
        data = type('Data', (), {'m': len(metadata['features'])})()
        
        model = Model(args, data)
        model.load_state_dict(state_dict)
        model.eval()
        
        return model, metadata
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

def create_scaler(data):
    scaler = MinMaxScaler()
    scaler.fit(data.reshape(-1, data.shape[-1]))
    return scaler

def preprocess_input(data, model_features, input_features, window_size, highway_window, scaler, h):
    feature_indices = [input_features.index(feat) for feat in model_features if feat in input_features]
    data_reordered = data[:, :, feature_indices]
    
    data_reshaped = data_reordered.reshape(-1, data_reordered.shape[-1])
    
    if scaler.n_features_in_ != data_reshaped.shape[1]:
        scaler.fit(data_reshaped)
    
    data_scaled = scaler.transform(data_reshaped).reshape(data_reordered.shape)
    
    if data_scaled.shape[1] > window_size:
        input_sequence = data_scaled[:, -window_size:, :]
    elif data_scaled.shape[1] < window_size:
        pad_length = window_size - data_scaled.shape[1]
        padding = np.zeros((data_scaled.shape[0], pad_length, data_scaled.shape[2]))
        input_sequence = np.concatenate([padding, data_scaled], axis=1)
    else:
        input_sequence = data_scaled
    
    h = min(h, window_size - highway_window)
    
    return torch.FloatTensor(input_sequence), h

def make_predictions(model, input_sequence, h, horizon):
    with torch.no_grad():
        predictions = []
        for i in range(input_sequence.shape[0]):
            window = input_sequence[i].unsqueeze(0)
            pred = model(window)
            predictions.append(pred.squeeze().numpy())
        predictions = np.array(predictions)
    
    return predictions

def process_predictions(predictions, timestamps, h, strategy='weighted_average', lambda_param=0.1):
    forecast_timestamps = timestamps[h:]
    
    if len(forecast_timestamps) == 0:
        h = max(0, len(timestamps) - 1)
        forecast_timestamps = timestamps[h:]
    
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    
    predictions_df = pd.DataFrame(predictions, columns=[f'h{i+1}' for i in range(predictions.shape[1])])
    predictions_df['timestamp'] = forecast_timestamps[:len(predictions_df)]
    
    melted = predictions_df.melt(id_vars=['timestamp'], var_name='horizon', value_name='prediction')
    melted['hour_offset'] = melted['horizon'].str.extract('(\d+)').astype(int)
    melted['pred_timestamp'] = melted['timestamp'] + pd.to_timedelta(melted['hour_offset'], unit='h')
    
    if strategy == 'average':
        return melted.groupby('pred_timestamp')['prediction'].mean().reset_index()
    elif strategy == 'most_recent':
        return melted.sort_values('timestamp').groupby('pred_timestamp').last().reset_index()
    elif strategy == 'weighted_average':
        melted['weight'] = np.exp(-lambda_param * melted['hour_offset'])
        weighted_avg = melted.groupby('pred_timestamp').apply(
            lambda x: np.average(x['prediction'], weights=x['weight'])
        ).reset_index(name='prediction')
        return weighted_avg
    else:
        return melted[melted['horizon'] == 'h1'][['pred_timestamp', 'prediction']]

def inverse_transform_predictions(predictions_df, scaler, features, target_feature):
    target_index = features.index(target_feature)
    dummy = np.zeros((len(predictions_df), scaler.n_features_in_))
    dummy[:, 0] = predictions_df['prediction'].values
    unscaled = scaler.inverse_transform(dummy)
    unscaled_predictions = unscaled[:, 0]
    
    return pd.DataFrame({
        'timestamp': predictions_df['pred_timestamp'],
        'prediction': unscaled_predictions
    })

def main():
    args = parse_args()
    
    try:
        model, metadata = load_model_from_s3(args.model_path)
        if model is None or metadata is None:
            raise ValueError("Failed to load model or metadata")
        
        input_data = load_from_s3(args.input_data)
        
        model_features = metadata['features']
        window_size = metadata['args']['window']
        highway_window = metadata['args']['highway_window']
        horizon = metadata['args']['horizon']
        h = input_data['h']
        
        scaler = create_scaler(input_data['X'])
        input_sequence, h = preprocess_input(input_data['X'], model_features, input_data['features'], 
                                             window_size, highway_window, scaler, h)
        
        raw_predictions = make_predictions(model, input_sequence, h, horizon)

        start_datetime = pd.to_datetime(input_data['start_datetime'])
        prediction_timestamps = [start_datetime + pd.Timedelta(hours=i) for i in range(len(input_sequence))]
        
        processed_predictions = process_predictions(raw_predictions, prediction_timestamps, h, args.strategy, args.lambda_param)
        
        if processed_predictions is None:
            raise ValueError("Failed to process predictions")
        
        target_feature = input_data['target']
        final_predictions = inverse_transform_predictions(processed_predictions, scaler, input_data['features'], target_feature)
        
        save_to_s3(final_predictions, args.output)
        print(f"Predictions saved to {args.output}")

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()