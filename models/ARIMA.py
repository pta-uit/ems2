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