FROM python:3.10-slim

WORKDIR /solar-app

COPY . .

RUN pip install --progress-bar off -r requirements.txt

ENTRYPOINT ["streamlit" , "run", "solar_app.py"]