ARG REGION=us-west-2
FROM 763104351884.dkr.ecr.${REGION}.amazonaws.com/pytorch-training:2.8.0-cpu-py312-ubuntu22.04-sagemaker-v1.0

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src

ENTRYPOINT ["python", "-m", "src.train"]