import argparse
import pickle
import numpy as np
import torch
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from models.LSTNet import Model
from sklearn.model_selection import train_test_split
from s3fs.core import S3FileSystem
import json
import os

print(f"PyTorch version: {torch.__version__}")

def get_s3(s3_path):
    s3 = S3FileSystem()
    return np.load(s3.open(s3_path), allow_pickle=True)

def load_s3(s3_path, arr):
    s3 = S3FileSystem()
    with s3.open(s3_path, 'wb') as f:
        f.write(pickle.dumps(arr))

def objective(params, X, y, device):
    class Data:
        def __init__(self, m):
            self.m = m

    data = Data(X.shape[2])

    # Convert relevant parameters to integers
    params['hidRNN'] = int(params['hidRNN'])
    params['hidCNN'] = int(params['hidCNN'])
    params['hidSkip'] = int(params['hidSkip'])
    params['CNN_kernel'] = int(params['CNN_kernel'])
    params['highway_window'] = int(params['highway_window'])
    params['skip'] = int(params['skip'])

    class Args:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    args = Args(**params)
    args.cuda = torch.cuda.is_available()
    args.window = X.shape[1]
    args.horizon = y.shape[1]
    args.output_fun = 'sigmoid'

    print(f"Data shapes: X: {X.shape}, y: {y.shape}")
    print(f"Args: {vars(args)}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert numpy arrays to PyTorch tensors
    try:
        X_train = torch.from_numpy(X_train).float().to(device)
        y_train = torch.from_numpy(y_train).float().to(device)
        X_val = torch.from_numpy(X_val).float().to(device)
        y_val = torch.from_numpy(y_val).float().to(device)
    except Exception as e:
        print(f"Error in tensor conversion: {e}")
        return {'loss': float('inf'), 'status': STATUS_OK}

    print(f"Tensor shapes: X_train: {X_train.shape}, y_train: {y_train.shape}")

    try:
        model = Model(args, data).to(device)
    except Exception as e:
        print(f"Error in model initialization: {e}")
        print(f"Args: {vars(args)}")
        print(f"Data shape: {data.m}")
        return {'loss': float('inf'), 'status': STATUS_OK}

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_epochs = 50  # You can adjust this
    for epoch in range(n_epochs):
        model.train()
        for i in range(0, len(X_train), args.batch_size):
            batch_X = X_train[i:i+args.batch_size]
            batch_y = y_train[i:i+args.batch_size]

            optimizer.zero_grad()
            try:
                outputs = model(batch_X)
            except Exception as e:
                print(f"Error in forward pass: {e}")
                print(f"batch_X shape: {batch_X.shape}")
                return {'loss': float('inf'), 'status': STATUS_OK}
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

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
    return {'loss': avg_val_loss, 'status': STATUS_OK}

def process_best_params(best, space):
    best_params = {}
    for key, value in best.items():
        if key in ['hidRNN', 'hidCNN', 'hidSkip', 'CNN_kernel', 'highway_window']:
            best_params[key] = int(value)
        elif key == 'skip':
            best_params[key] = space['skip'][value]
        elif key == 'batch_size':
            best_params[key] = space['batch_size'][value]
        elif key in ['dropout', 'lr']:
            best_params[key] = value
        else:
            best_params[key] = value
    return best_params

def validate_params(params):
    params['hidRNN'] = max(50, min(200, int(params['hidRNN'])))
    params['hidCNN'] = max(50, min(200, int(params['hidCNN'])))
    params['hidSkip'] = max(1, min(10, int(params['hidSkip'])))
    params['CNN_kernel'] = max(3, min(8, int(params['CNN_kernel'])))
    params['skip'] = 24 if params['skip'] not in [12, 24, 48] else params['skip']
    params['highway_window'] = max(12, min(48, int(params['highway_window'])))
    params['dropout'] = max(0.1, min(0.5, params['dropout']))
    params['lr'] = max(1e-4, min(1e-2, params['lr']))
    params['batch_size'] = 64 if params['batch_size'] not in [32, 64, 128, 256] else params['batch_size']
    return params

def run_hyperparameter_tuning(preprocessed_data_path, max_evals=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        preprocessed_data = get_s3(preprocessed_data_path)
    except Exception as e:
        print(f"Error loading preprocessed data: {e}")
        return None

    try:
        X = np.array(preprocessed_data['X'])
        y = np.array(preprocessed_data['y'])
    except Exception as e:
        print(f"Error converting data to numpy arrays: {e}")
        return None

    print(f"Data loaded. X shape: {X.shape}, y shape: {y.shape}")

    space = {
        'hidRNN': hp.quniform('hidRNN', 50, 200, 1),
        'hidCNN': hp.quniform('hidCNN', 50, 200, 1),
        'hidSkip': hp.quniform('hidSkip', 1, 10, 1),
        'CNN_kernel': hp.quniform('CNN_kernel', 3, 8, 1),
        'skip': hp.choice('skip', [12, 24, 48]),
        'highway_window': hp.quniform('highway_window', 12, 48, 1),
        'dropout': hp.uniform('dropout', 0.1, 0.5),
        'lr': hp.loguniform('lr', np.log(1e-4), np.log(1e-2)),
        'batch_size': hp.choice('batch_size', [32, 64, 128, 256])
    }

    trials = Trials()
    try:
        best = fmin(fn=lambda params: objective(params, X, y, device),
                    space=space,
                    algo=tpe.suggest,
                    max_evals=max_evals,
                    trials=trials)
    except Exception as e:
        print(f"Error during hyperparameter optimization: {e}")
        return None

    try:
        best_params = process_best_params(best, space)
        best_params = validate_params(best_params)
    except Exception as e:
        print(f"Error processing best parameters: {e}")
        return None

    print('Best hyperparameters:')
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    return best_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning for LSTNet Solar Generation Forecasting')
    parser.add_argument('--preprocessed_data', type=str, required=True, help='S3 path to the preprocessed data file')
    parser.add_argument('--max_evals', type=int, default=100, help='Maximum number of evaluations for hyperparameter tuning')
    parser.add_argument('--save', type=str, required=True, help='S3 path to save the best hyperparameters')
    args = parser.parse_args()

    try:
        best_params = run_hyperparameter_tuning(args.preprocessed_data, args.max_evals)
        if best_params is None:
            raise ValueError("Hyperparameter tuning failed to return valid parameters.")
        
        best_params = validate_params(best_params)
        print(f'Best hyperparameters: {best_params}')

        # Save best parameters to S3 as JSON
        s3 = S3FileSystem()
        with s3.open(os.path.join(args.save, "best_params.json"), 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f'Saved best hyperparameters to {args.save}')
    except Exception as e:
        print(f"Error in hyperparameter tuning: {e}")
        print("Hyperparameter tuning failed. No parameters were saved.")
        print("The main script will use default parameters.")