import argparse
import torch
import numpy as np
import pandas as pd
from s3fs.core import S3FileSystem
from models.LSTNet import Model as LSTNet
from models.attention_lstm import AttentionLSTM
import os
from sklearn.preprocessing import MinMaxScaler
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='Make predictions using deep learning models')
    parser.add_argument('--model_path', type=str, required=True, help='S3 path to the model directory')
    parser.add_argument('--model', type=str, required=True, choices=['lstnet', 'attention'], 
                       help='Model type to use for prediction')
    parser.add_argument('--input_data', type=str, required=True, help='S3 path to the prepared input data pickle file')
    parser.add_argument('--output', type=str, required=True, help='S3 path to save predictions directory')
    parser.add_argument('--strategy', type=str, default='weighted_average', 
                       choices=['single', 'average', 'most_recent', 'weighted_average'], 
                       help='Strategy for processing predictions')
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

def get_model_paths(base_path, model_type):
    """Get model and metadata paths based on model type."""
    model_filename = f"{model_type}_model.pt"
    metadata_filename = f"{model_type}_model_metadata.pt"
    
    model_path = os.path.join(base_path, model_filename)
    metadata_path = os.path.join(base_path, metadata_filename)
    
    return model_path, metadata_path

def get_model_class(model_type):
    """Get the appropriate model class based on model type."""
    if model_type == 'lstnet':
        return LSTNet
    elif model_type == 'attention':
        return AttentionLSTM
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def load_model_from_s3(model_path, metadata_path, model_type):
    try:
        state_dict = load_from_s3(model_path)
        metadata = load_from_s3(metadata_path)
        
        # Add missing attributes required by both models
        args_dict = metadata['args']
        if 'cuda' not in args_dict:
            args_dict['cuda'] = False
        
        # Create Namespace with the arguments
        args = argparse.Namespace(**args_dict)
        
        # Create data object with number of features
        data = type('Data', (), {'m': len(metadata['features'])})()
        
        # Get appropriate model class and initialize
        ModelClass = get_model_class(model_type)
        model = ModelClass(args, data)
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
    
    if highway_window is not None:  # Only for LSTNet
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
    dummy[:, target_index] = predictions_df['prediction'].values
    unscaled = scaler.inverse_transform(dummy)
    unscaled_predictions = unscaled[:, target_index]
    
    return pd.DataFrame({
        'timestamp': predictions_df['pred_timestamp'],
        'prediction': unscaled_predictions
    })

def main():
    args = parse_args()
    
    try:
        # Get model and metadata paths
        model_path, metadata_path = get_model_paths(args.model_path, args.model)
        
        # Load model and metadata
        model, metadata = load_model_from_s3(model_path, metadata_path, args.model)
        if model is None or metadata is None:
            raise ValueError("Failed to load model or metadata")
        
        # Load input data
        input_data = load_from_s3(args.input_data)
        
        # Get model parameters
        model_features = metadata['features']
        window_size = metadata['args']['window']
        highway_window = metadata['args'].get('highway_window')  # Only for LSTNet
        horizon = metadata['args']['horizon']
        h = input_data['h']
        
        # Prepare input data
        scaler = create_scaler(input_data['X'])
        input_sequence, h = preprocess_input(input_data['X'], model_features, input_data['features'], 
                                           window_size, highway_window, scaler, h)
        
        # Make predictions
        raw_predictions = make_predictions(model, input_sequence, h, horizon)
        
        # Process timestamps
        start_datetime = pd.to_datetime(input_data['start_datetime'])
        prediction_timestamps = [start_datetime + pd.Timedelta(hours=i) for i in range(len(input_sequence))]
        
        # Process predictions
        processed_predictions = process_predictions(raw_predictions, prediction_timestamps, h, 
                                                 args.strategy, args.lambda_param)
        
        if processed_predictions is None:
            raise ValueError("Failed to process predictions")
        
        # Transform predictions back to original scale
        target_feature = input_data['target']
        final_predictions = inverse_transform_predictions(processed_predictions, scaler, 
                                                       input_data['features'], target_feature)
        
        # Save predictions
        output_path = os.path.join(args.output, f"{args.model}_predictions.csv")
        save_to_s3(final_predictions, output_path)
        print(f"Predictions saved to {output_path}")

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()