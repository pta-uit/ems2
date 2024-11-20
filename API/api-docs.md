# solar-prediction-api.py

## 1. Requirements

```txt
Flask==3.0.3
pandas==2.2.2
s3fs==2024.9.0
boto3==1.35.39
```
- access permission to S3 bucket
## 2. Usage

```bash
python3 solar-prediction-api.py --predictions s3://bucket/file.csv
```

```bash
python3 solar-prediction-api.py --predictions s3://trambk/solar-energy/model/predictions.csv
```

Endpoint at "http://localhost:5000/predict"

## 3. Test

```bash
curl "http://localhost:5000/predict?datetime=2022-05-25T00:00:00&n_hours=24"
```

The response should be:
```json
[
  {
    "prediction": 0.3534320310145835,
    "timestamp": "Wed, 25 May 2022 01:00:00 GMT"
  },
  {
    "prediction": 0.353425952504915,
    "timestamp": "Wed, 25 May 2022 02:00:00 GMT"
  },
  {
    "prediction": 0.3534363528772228,
    "timestamp": "Wed, 25 May 2022 03:00:00 GMT"
  },
  {
    "prediction": 0.3534732936475341,
    "timestamp": "Wed, 25 May 2022 04:00:00 GMT"
  },

...
]
```

# solar_app.py

## 1. Requirements

```txt
streamlit==1.39.0
pandas==2.2.2
requests==2.32.3
plotly==5.24.1
```

Make sure the api server available

## 2. Usage

```bash
streamlit run solar_app.py
```

Select available value for fields, currently 2022/05/25, 00:00, 24.