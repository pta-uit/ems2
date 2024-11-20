## DOCKER

**1. Solar prediction**
- Configure AWS credentials that has permission to access S3 bucket in advance so that you can mount the credentials to the container
```bash
cd solar-prediction
docker build -t solar-prediction-api .
docker run -v $HOME/.aws/credentials:/root/.aws/credentials:ro -p 5000:5000 -it solar-prediction-api --predictions s3://trambk/solar-energy/model/predictions.csv
```

**2. Solar app**
```bash
cd solar-app
docker build -t solar-app .
docker run -p 8501:8501 solar-app
```

## KUBERNETES

- First, modify the code in **solar_app.py** to point the API URL to use cluster IP DNS instead (localhost:5000 => solar-prediction-api-service:5000)

- Prepare secrets (ecr-secret and aws secret) so that kubernetes can pull images from ECR. When create the aws-secret, for easy implementation, direct to the folder that contains your aws credentials file (typically $HOME/.aws)
```bash
kubectl create secret docker-registry ecr-secret --docker-server=<your-account-id>.dkr.ecr.ap-southeast-1.amazonaws.com --docker-username=AWS --docker-password=$(aws ecr get-login-password) --namespace=default

kubectl create secret generic aws-secret --from-file=credentials 
```
- Apply kubernetes deployment and service
```bash
kubectl apply -f api-server.yaml
kubectl apply -f solar-app.yaml

kubectl port-forward svc/solar-streamlit-service 8501:8501
```
Then you can access the streamlit UI via localhost:8501