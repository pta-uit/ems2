import argparse
import torch
import numpy as np
import pandas as pd
from models.LSTNet import Model as LSTNet
from models.ARIMA import ArimaModel
from sklearn.model_selection import train_test_split
import os
import json
from s3fs.core import S3FileSystem
import boto3
import mlflow
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='Time Series Forecasting with LSTNet/ARIMA')
    
    # Model selection argument
    parser.add_argument('--model_type', type=str, choices=['lstnet', 'arima'], default='lstnet',
                       help='Type of model to use (lstnet or arima)')
    
    # ARIMA-specific arguments with defaults
    parser.add_argument('--ar_order', type=int, default=2, help='AR order for ARIMA model (default: 2)')
    parser.add_argument('--diff_order', type=int, default=1, help='Differencing order for ARIMA model (default: 1)')
    parser.add_argument('--ma_order', type=int, default=2, help='MA order for ARIMA model (default: 2)')
    
    # Required arguments
    parser.add_argument('--preprocessed_data', type=str, required=True, help='Path to the preprocessed data file')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use (default: -1, i.e., CPU)')
    parser.add_argument('--save', type=str, default='model.pt', help='Path to save the model')
    
    # LSTNet-specific arguments (keeping your existing defaults)
    parser.add_argument('--hidRNN', type=int, default=100, help='Number of RNN hidden units (default: 100)')
    parser.add_argument('--hidCNN', type=int, default=100, help='Number of CNN hidden units (default: 100)')
    parser.add_argument('--hidSkip', type=int, default=5, help='Number of skip RNN hidden units (default: 5)')
    parser.add_argument('--CNN_kernel', type=int, default=6, help='CNN kernel size (default: 6)')
    parser.add_argument('--skip', type=int, default=24, help='Skip length (default: 24)')
    parser.add_argument('--highway_window', type=int, default=24, help='Highway window size (default: 24)')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (default: 0.2)')
    parser.add_argument('--output_fun', type=str, default='sigmoid', help='Output function: sigmoid, tanh or None (default: sigmoid)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--loss_history', type=str, default=None, help='Path to save the loss history')
    parser.add_argument('--mlflow_uri', type=str, default="http://localhost:5000")
    parser.add_argument('--best_params', type=str, default=None, help='S3 path to the best hyperparameters JSON file')
    
    args = parser.parse_args()
    args.cuda = args.gpu >= 0 and torch.cuda.is_available()
    return args

def train_lstnet(model, X_train, y_train, X_val, y_val, args, device):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    train_losses = []
    val_losses = []
    
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
    
    return train_losses, val_losses

def train_arima(model, X_train, y_train, X_val, y_val, args):
    # ARIMA doesn't need the same kind of epoch-based training
    print("Training ARIMA model...")
    model.fit(X_train, y_train)
    
    # Calculate training error
    train_pred = np.array([model.predict(None) for _ in range(len(y_train))])
    train_loss = np.mean((train_pred - y_train) ** 2)
    
    # Calculate validation error
    val_pred = np.array([model.predict(None) for _ in range(len(y_val))])
    val_loss = np.mean((val_pred - y_val) ** 2)
    
    print(f'Training MSE: {train_loss:.4f}, Validation MSE: {val_loss:.4f}')
    
    return [train_loss], [val_loss]

def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if args.cuda else 'cpu')
    
    # Load and prepare data
    preprocessed_data = get_s3(args.preprocessed_data)
    X = preprocessed_data['X']
    y = preprocessed_data['y']
    features = preprocessed_data['features']
    
    args.window = preprocessed_data['window']
    args.horizon = preprocessed_data['horizon']
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize model based on model_type
    if args.model_type == 'lstnet':
        # Prepare data for LSTNet
        X_train = torch.FloatTensor(X_train).to(device)
        y_train = torch.FloatTensor(y_train).to(device)
        X_val = torch.FloatTensor(X_val).to(device)
        y_val = torch.FloatTensor(y_val).to(device)
        
        data = type('Data', (), {'m': len(features)})()
        model = LSTNet(args, data).to(device)
        train_losses, val_losses = train_lstnet(model, X_train, y_train, X_val, y_val, args, device)
    
    else:  # arima
        model = ArimaModel(args)
        train_losses, val_losses = train_arima(model, X_train, y_train, X_val, y_val, args)
    
    # MLflow logging
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("solar-generation")
    
    with mlflow.start_run(run_name=f"{args.model_type}-solar-gen") as run:
        # Log common parameters
        mlflow.log_param("model_type", args.model_type)
        
        if args.model_type == 'lstnet':
            # Log LSTNet specific parameters
            mlflow.log_param("hidRNN", args.hidRNN)
            mlflow.log_param("hidCNN", args.hidCNN)
            mlflow.log_param("hidSkip", args.hidSkip)
            mlflow.log_param("CNN_kernel", args.CNN_kernel)
            mlflow.log_param("skip", args.skip)
            mlflow.log_param("highway_window", args.highway_window)
            mlflow.log_param("dropout", args.dropout)
            mlflow.log_param("output_fun", args.output_fun)
        else:
            # Log ARIMA specific parameters
            mlflow.log_param("ar_order", args.ar_order)
            mlflow.log_param("diff_order", args.diff_order)
            mlflow.log_param("ma_order", args.ma_order)
        
        # Log metrics
        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
        
        # Save model
        if args.model_type == 'lstnet':
            torch.save(model.state_dict(), "model.pt")
            mlflow.pytorch.log_model(model, "model")
        else:
            model.save("model.pkl")
            mlflow.pyfunc.log_model("model", python_model=model)
        
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri=model_uri, name=f"solar-gen-{args.model_type}")
        
        # Save to S3
        load_s3(os.path.join(args.save, f"model.{'pt' if args.model_type == 'lstnet' else 'pkl'}"),
                model.state_dict() if args.model_type == 'lstnet' else model)

if __name__ == "__main__":
    main()