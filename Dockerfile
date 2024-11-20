# Build stage
FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --user --no-cache-dir -r requirements.txt

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf aws awscliv2.zip

# Final stage
FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY --from=builder /usr/local/aws-cli /usr/local/aws-cli

ENV PATH=/root/.local/bin:/usr/local/aws-cli/v2/current/bin:$PATH
ENV PREPROCESSED_DATA=""
ENV MODEL_PATH=""
ENV BEST_PARAMS=""
ENV GPU=-1
ENV FLASK_ENV=production

COPY . .

EXPOSE 5000

# CMD ["sh", "-c", "python api.py --gpu $GPU" --preprocessed_data $PREPROCESSED_DATA 
#                                 --model_path $MODEL_PATH --hyperparams_path $BEST_PARAMS]
CMD ["sh", "-c", "python api.py"]