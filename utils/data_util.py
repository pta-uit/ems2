import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import STL

class DataUtil:
    def __init__(self, weather_file, building_file, year=None, solar=False):
        self.weather_file = weather_file
        self.building_file = building_file
        self.year = year or pd.Timestamp.now().year
        self.solar = solar
        self.df = None
        self.scaler = MinMaxScaler()

    def load_and_preprocess_data(self):
        # Load and merge weather and building data
        df_weather = pd.read_csv(self.weather_file)
        df_building = pd.read_csv(self.building_file)
        
        if self.solar:
            building_columns = ['Month', 'Hour', 'Solar Generation [W/kW]']
            df_building_selected = df_building[building_columns]
        else:
            df_building_selected = df_building[['Month', 'Hour']]
        
        df = pd.concat([df_building_selected, df_weather], axis=1)
        
        # Process datetime
        df = self._create_date_time_column(df)
        df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
        df.set_index('DATE_TIME', inplace=True)
        df.sort_index(inplace=True)
        
        self.df = df
        self._verify_data_integrity()
        
        return self.df

    def _create_date_time_column(self, df):
        df = df.copy()
        df['Day'] = 1
        df['cumulative_hours'] = df.groupby('Month').cumcount()
        days_per_month = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
        
        for month in df['Month'].unique():
            mask = df['Month'] == month
            df.loc[mask, 'Day'] = (df.loc[mask, 'cumulative_hours'] // 24) + 1
            df.loc[mask, 'Day'] = df.loc[mask, 'Day'].clip(upper=days_per_month[month])

        df['DATE_TIME'] = pd.to_datetime(dict(year=self.year, month=df['Month'], day=df['Day'], hour=df['Hour']))
        df = df.sort_values('DATE_TIME')
        df = df.drop(columns=['cumulative_hours'])
        cols = ['DATE_TIME'] + [col for col in df if col != 'DATE_TIME']
        return df[cols]

    def _verify_data_integrity(self):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_preprocess_data() first.")

        missing_values = self.df.isnull().sum()
        if missing_values.any():
            raise ValueError(f"Missing values detected: {missing_values[missing_values > 0]}")
        
        duplicates = self.df.index.duplicated()
        if duplicates.any():
            raise ValueError(f"{duplicates.sum()} duplicate timestamps detected")
        
        time_diff = self.df.index.to_series().diff()
        large_gaps = time_diff[time_diff > pd.Timedelta(hours=1)]
        if not large_gaps.empty:
            raise ValueError(f"Large gaps detected in the time series: {large_gaps}")

    def verify_preprocessing(self):
        if self.df is None:
            raise ValueError("Data has not been loaded and preprocessed yet.")

        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise ValueError("DATE_TIME is not set as the index.")

        if not self.df.index.is_monotonic_increasing:
            raise ValueError("Data is not sorted in ascending order.")

        expected_columns = ['Outdoor Drybulb Temperature [C]', 'Relative Humidity [%]', 
                            'Diffuse Solar Radiation [W/m2]', 'Direct Solar Radiation [W/m2]']
        if not all(col in self.df.columns for col in expected_columns):
            raise ValueError("Some expected columns are missing.")

        if self.solar and 'Solar Generation [W/kW]' not in self.df.columns:
            raise ValueError("Solar Generation [W/kW] column is missing.")

        time_diff = self.df.index.to_series().diff()
        if not (time_diff[1:] == pd.Timedelta(hours=1)).all():
            raise ValueError("Time intervals are not consistently 1 hour.")

        return True

    def perform_feature_engineering(self):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_preprocess_data() first.")

        # Create time-based features
        self.df['hour'] = self.df.index.hour
        self.df['day_of_year'] = self.df.index.dayofyear
        self.df['month'] = self.df.index.month

        self.df['hour_sin'] = np.sin(self.df['hour'] * (2 * np.pi / 24))
        self.df['hour_cos'] = np.cos(self.df['hour'] * (2 * np.pi / 24))
        self.df['day_of_year_sin'] = np.sin(self.df['day_of_year'] * (2 * np.pi / 365))
        self.df['day_of_year_cos'] = np.cos(self.df['day_of_year'] * (2 * np.pi / 365))

        # Perform STL decomposition
        solar_column = 'Solar Generation [W/kW]' if 'Solar Generation [W/kW]' in self.df.columns else 'Direct Solar Radiation [W/m2]'
        stl_series = self.df[solar_column]
        result = STL(stl_series, period=24).fit()
        self.df['trend'], self.df['seasonal'], self.df['residual'] = result.trend, result.seasonal, result.resid

        return self.df

    def prepare_sequences(self, window_size, horizon, features, target=None):
        if self.df is None:
            raise ValueError("Data not loaded and preprocessed. Call load_and_preprocess_data() and perform_feature_engineering() first.")

        if target and target in self.df.columns:
            data = self.scaler.fit_transform(self.df[features + [target]])
            X = np.array([data[i:(i+window_size), :-1] for i in range(len(data) - window_size - horizon + 1)])
            y = np.array([data[(i+window_size):(i+window_size+horizon), -1] for i in range(len(data) - window_size - horizon + 1)])
            return X, y
        else:
            data = self.scaler.fit_transform(self.df[features])
            X = np.array([data[i:(i+window_size), :] for i in range(len(data) - window_size + 1)])
            return X, None

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)