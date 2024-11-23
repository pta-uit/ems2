import argparse
import torch
import numpy as np
from models.LSTNet import Model as LSTNet
from models.attention_lstm import AttentionLSTM
from sklearn.model_selection import train_test_split
import os
import json
from s3fs.core import S3FileSystem
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='Deep Learning Models for Solar Generation Forecasting')
    
    # Common arguments
    parser.add_argument('--model', type=str, required=True, choices=['lstnet', 'attention'],
                       help='Model to use: lstnet or attention')
    parser.add_argument('--preprocessed_data', type=str, required=True, help='Path to the preprocessed data file')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use (default: -1, i.e., CPU)')
    parser.add_argument('--save', type=str, default='model.pt', help='Path to save the model')
    parser.add_argument('--hidRNN', type=int, default=100, help='Number of RNN hidden units (default: 100)')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (default: 0.2)')
    parser.add_argument('--output_fun', type=str, default='sigmoid', help='Output function: sigmoid, tanh or None')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--loss_history', type=str, default=None, help='Path to save the loss history')
    parser.add_argument('--best_params', type=str, default=None, help='S3 path to the best hyperparameters JSON file')
    
    # Create argument groups for model-specific parameters
    lstnet_group = parser.add_argument_group('LSTNet specific arguments')
    lstnet_group.add_argument('--hidCNN', type=int, default=100, help='Number of CNN hidden units')
    lstnet_group.add_argument('--hidSkip', type=int, default=5, help='Number of skip RNN hidden units')
    lstnet_group.add_argument('--CNN_kernel', type=int, default=6, help='CNN kernel size')
    lstnet_group.add_argument('--skip', type=int, default=24, help='Skip length')
    lstnet_group.add_argument('--highway_window', type=int, default=24, help='Highway window size')
    
    attention_group = parser.add_argument_group('Attention LSTM specific arguments')
    attention_group.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    
    args = parser.parse_args()
    args.cuda = args.gpu >= 0 and torch.cuda.is_available()
    return args

def save_s3(s3_path,arr):
    s3 = S3FileSystem()
    with s3.open(s3_path, 'wb') as f:
        f.write(pickle.dumps(arr))

def get_s3(s3_path):
    s3 = S3FileSystem()
    return np.load(s3.open(s3_path), allow_pickle=True)

def save_model(model, save_path, args, features):
    """
    Save model and filtered metadata to S3 bucket.
    Ensures all necessary parameters are saved for model loading and prediction.
    """
    # Save model state dict
    model_state = model.state_dict()
    save_s3(save_path, model_state)
    
    # Filter parameters based on model type and save metadata
    filtered_params = filter_model_params(args, args.model)
    metadata = {
        'args': filtered_params,
        'features': features
    }
    metadata_path = save_path.replace('.pt', '_metadata.pt')
    save_s3(metadata_path, metadata)

    print(f'Model saved to S3: {save_path}')
    print(f'Filtered metadata saved to S3: {metadata_path}')
    
def get_model(args, data):
    if args.model == 'lstnet':
        return LSTNet(args, data)
    else:  # attention
        return AttentionLSTM(args, data)
    
def filter_model_params(args, model_type):
    """
    Filter and return only the relevant parameters for the specified model type,
    ensuring all necessary parameters are included for model loading and prediction.
    """
    # Common parameters required for all models
    common_params = {
        'model': args.model,
        'preprocessed_data': args.preprocessed_data,
        'gpu': args.gpu,
        'cuda': args.cuda,  # Essential for model loading
        'hidRNN': args.hidRNN,
        'dropout': args.dropout,
        'output_fun': args.output_fun,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'window': args.window,
        'horizon': args.horizon
    }
    
    if model_type == 'lstnet':
        # Add LSTNet specific parameters
        model_params = {
            **common_params,
            'hidCNN': args.hidCNN,
            'hidSkip': args.hidSkip,
            'CNN_kernel': args.CNN_kernel,
            'skip': args.skip,
            'highway_window': args.highway_window
        }
    elif model_type == 'attention':
        # Add Attention LSTM specific parameters
        model_params = {
            **common_params,
            'num_layers': args.num_layers
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return model_params

def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if args.cuda else 'cpu')
    print(f"Using device: {device}")
    print(f"Training with {args.model.upper()} model")

    # Load data and prepare datasets
    preprocessed_data = get_s3(args.preprocessed_data)
    X = preprocessed_data['X']
    y = preprocessed_data['y']
    features = preprocessed_data['features']
    
    args.window = preprocessed_data['window']
    args.horizon = preprocessed_data['horizon']
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.FloatTensor(y_val).to(device)

    # Create a Data object
    class Data:
        def __init__(self, m):
            self.m = m

    data = Data(len(features))

    # Initialize the selected model
    model = get_model(args, data).to(device)
    print(f"Initialized {args.model.upper()} model with {sum(p.numel() for p in model.parameters())} parameters")

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for i in range(0, len(X_train), args.batch_size):
            batch_X = X_train[i:i+args.batch_size]
            batch_y = y_train[i:i+args.batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (len(X_train) / args.batch_size)
        train_losses.append(avg_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i in range(0, len(X_val), args.batch_size):
                batch_X = X_val[i:i+args.batch_size]
                batch_y = y_val[i:i+args.batch_size]
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / (len(X_val) / args.batch_size)
        val_losses.append(avg_val_loss)

        print(f'Epoch [{epoch+1}/{args.epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    # Save model and history
    save_path = os.path.join(args.save, f"{args.model}_model.pt")
    save_model(model, save_path, args, features)

    if args.loss_history:
        history = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'model_type': args.model
        }
        history_path = args.loss_history.replace('.json', f'_{args.model}.json')
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        with open(history_path, 'w') as f:
            json.dump(history, f)
        print(f'Loss history saved to {history_path}')

if __name__ == "__main__":
    main()