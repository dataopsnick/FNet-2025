ARG REGION=us-west-2
FROM 763104351884.dkr.ecr.${REGION}.amazonaws.com/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-sagemaker

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m nltk.downloader punkt wordnet

COPY src/ /app/src
COPY src/ds_config.json /app/src/ds_config.json

ENTRYPOINT ["python", "-m", "src.train"]