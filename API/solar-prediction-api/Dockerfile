FROM python:3.10-slim AS builder

WORKDIR /solar-prediction

COPY requirements.txt ./

RUN pip install -r requirements.txt

#########################
#########################

FROM python:3.10-slim

WORKDIR /solar-prediction

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

COPY . .

RUN pip install awscli

ENTRYPOINT ["python", "solar-prediction-api.py"]