import pandas as pd
import numpy as np
import argparse
from s3fs.core import S3FileSystem
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_s3_csv(s3_path):
    s3 = S3FileSystem()
    with s3.open(s3_path, 'r') as f:
        return pd.read_csv(f)

def save_s3_csv(df, s3_path):
    s3 = S3FileSystem()
    with s3.open(s3_path, 'w') as f:
        df.to_csv(f, index=False)

def validate_and_clean_data(weather_file, building_file):
    """Validate and attempt to clean the raw input data."""
    df_weather = load_s3_csv(weather_file)
    df_building = load_s3_csv(building_file)
    
    issues = {"weather": {}, "building": {}}
    corrections = {"weather": {}, "building": {}}

    # Check for missing values
    for df_name, df in [("weather", df_weather), ("building", df_building)]:
        missing = df.isnull().sum()
        if missing.sum() > 0:
            issues[df_name]["missing_values"] = missing[missing > 0].to_dict()
            # Attempt to fill missing values with mean of the column
            df.fillna(df.mean(), inplace=True)
            corrections[df_name]["filled_missing_values"] = missing[missing > 0].to_dict()

    # Check for negative values in solar radiation and temperature
    negative_weather = (df_weather[['Outdoor Drybulb Temperature [C]', 'Diffuse Solar Radiation [W/m2]', 'Direct Solar Radiation [W/m2]']] < 0).sum()
    if negative_weather.sum() > 0:
        issues["weather"]["negative_values"] = negative_weather[negative_weather > 0].to_dict()
        # Set negative values to 0
        for col in negative_weather.index:
            df_weather.loc[df_weather[col] < 0, col] = 0
        corrections["weather"]["corrected_negative_values"] = negative_weather[negative_weather > 0].to_dict()

    # Check for out-of-range relative humidity
    humidity_issues = ((df_weather['Relative Humidity [%]'] < 0) | (df_weather['Relative Humidity [%]'] > 100)).sum()
    if humidity_issues > 0:
        issues["weather"]["out_of_range_humidity"] = humidity_issues
        # Clip humidity values to [0, 100] range
        df_weather['Relative Humidity [%]'] = df_weather['Relative Humidity [%]'].clip(0, 100)
        corrections["weather"]["clipped_humidity_values"] = humidity_issues

    # Check for negative solar generation
    negative_generation = (df_building['Solar Generation [W/kW]'] < 0).sum()
    if negative_generation > 0:
        issues["building"]["negative_generation"] = negative_generation
        # Set negative generation to 0
        df_building.loc[df_building['Solar Generation [W/kW]'] < 0, 'Solar Generation [W/kW]'] = 0
        corrections["building"]["corrected_negative_generation"] = negative_generation

    return df_weather, df_building, issues, corrections

def main():
    parser = argparse.ArgumentParser(description='Validate and clean raw data for Solar Generation Forecasting')
    parser.add_argument('--weather_data', type=str, required=True, help='S3 path to the weather CSV file')
    parser.add_argument('--building_data', type=str, required=True, help='S3 path to the building CSV file')
    parser.add_argument('--output_report', type=str, required=True, help='S3 path to save the validation report')
    args = parser.parse_args()

    logger.info("Validating and cleaning raw data...")
    df_weather, df_building, issues, corrections = validate_and_clean_data(args.weather_data, args.building_data)

    # Save cleaned data
    save_s3_csv(df_weather, args.weather_data.replace('.csv', '_cleaned.csv'))
    save_s3_csv(df_building, args.building_data.replace('.csv', '_cleaned.csv'))

    # Prepare and save report
    report = {
        "issues_found": issues,
        "corrections_made": corrections,
        "cleaned_weather_data": args.weather_data.replace('.csv', '_cleaned.csv'),
        "cleaned_building_data": args.building_data.replace('.csv', '_cleaned.csv')
    }

    s3 = S3FileSystem()
    with s3.open(args.output_report, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Data validation and cleaning complete. Report saved to {args.output_report}")

    if any(issues.values()):
        logger.warning("Issues were found and corrected. Please review the report for details.")
    else:
        logger.info("No issues found in the raw data.")

if __name__ == "__main__":
    main()