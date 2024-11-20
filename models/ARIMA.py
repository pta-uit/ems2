from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class ArimaModel:
    def __init__(self, args):
        """
        Initialize ARIMA model with default orders (2,1,2)
        Args:
            args: arguments containing ARIMA parameters
        """
        self.order = (
            getattr(args, 'ar_order', 2),  # default p=2
            getattr(args, 'diff_order', 1), # default d=1
            getattr(args, 'ma_order', 2)    # default q=2
        )
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
        # We'll use only the last value from each window
        return data[:, -1, :]
    
    def fit(self, X, y):
        """
        Fit ARIMA model for each target variable
        Args:
            X: input data of shape (n_samples, window_size, n_features)
            y: target values of shape (n_samples, horizon)
        """
        print(f"Fitting ARIMA model with order={self.order}")
        
        # Reshape input data to use the last timestep of each window
        train_data = self.reshape_data(X)
        
        self.models = []
        # Fit separate ARIMA model for each horizon step
        for i in range(y.shape[1]):
            model = ARIMA(train_data[:, i], order=self.order)
            fitted_model = model.fit()
            self.models.append(fitted_model)
            print(f"Fitted model for horizon step {i+1}")
            
    def predict(self, X):
        """
        Generate predictions
        Args:
            X: input data of shape (n_samples, window_size, n_features)
        Returns:
            predictions: array of shape (n_samples, horizon)
        """
        if self.models is None:
            raise ValueError("Model needs to be fitted before making predictions")
            
        # Reshape input data
        test_data = self.reshape_data(X)
        
        predictions = np.zeros((len(test_data), len(self.models)))
        
        # Generate predictions for each horizon step
        for i, model in enumerate(self.models):
            forecast = model.forecast(steps=1)
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