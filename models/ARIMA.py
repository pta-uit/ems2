from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class ArimaModel:
    def __init__(self, args):
        """
        Initialize ARIMA model with optimized parameters for solar generation
        Args:
            args: arguments containing ARIMA parameters
        """
        # Optimized parameters for solar generation:
        # p=24 to capture daily patterns
        # d=1 for first-order differencing to achieve stationarity
        # q=1 for simple moving average component
        self.order = (24, 1, 1)
        self.horizon = args.horizon
        self.models = None
        
    def reshape_data(self, data):
        """
        Reshape the windowed data into a format suitable for ARIMA
        Args:
            data: numpy array of shape (n_samples, window_size, n_features)
        Returns:
            reshaped_data: numpy array of shape (n_samples, n_features)
        """
        if data is None:
            raise ValueError("Input data cannot be None")
        return data[:, -1, :]
    
    def fit(self, X, y):
        """
        Fit ARIMA model for solar generation forecasting
        Args:
            X: input data of shape (n_samples, window_size, n_features)
            y: target values of shape (n_samples, horizon)
        """
        print(f"Fitting ARIMA model with order={self.order}")
        
        # Reshape input data to use the last timestep of each window
        train_data = self.reshape_data(X)
        
        self.models = []
        # Use the target variable (solar generation) for training
        target_data = y
        
        # Fit separate ARIMA model for each horizon step
        for i in range(self.horizon):
            model = ARIMA(target_data[:, i], order=self.order)
            fitted_model = model.fit()
            self.models.append(fitted_model)
            print(f"Fitted model for horizon step {i+1}")
            
    def predict(self, X):
        """
        Generate predictions for solar generation
        Args:
            X: input data of shape (n_samples, window_size, n_features)
        Returns:
            predictions: array of shape (n_samples, horizon)
        """
        if self.models is None:
            raise ValueError("Model needs to be fitted before making predictions")
        
        predictions = np.zeros((X.shape[0], self.horizon))
        
        # Generate predictions for each horizon step
        for i, model in enumerate(self.models):
            forecast = model.forecast(steps=X.shape[0])
            predictions[:, i] = forecast
            
        return predictions
    
    def evaluate(self, X, y):
        """
        Evaluate the model's performance
        Args:
            X: input data
            y: true values
        Returns:
            mse: Mean squared error
        """
        predictions = self.predict(X)
        # Ensure predictions are non-negative (since solar generation can't be negative)
        predictions = np.maximum(predictions, 0)
        mse = mean_squared_error(y, predictions)
        return mse
    
    def save(self, path):
        """Save the ARIMA models"""
        with open(path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'order': self.order,
                'horizon': self.horizon
            }, f)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load the ARIMA models"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.models = data['models']
            self.order = data['order']
            self.horizon = data['horizon']
        print(f"Model loaded from {path}")