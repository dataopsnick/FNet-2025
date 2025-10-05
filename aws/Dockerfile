ARG REGION=us-west-2
FROM 763104351884.dkr.ecr.${REGION}.amazonaws.com/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-sagemaker

# Install git for cloning the repository
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install base requirements. These will be cached by Docker.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m nltk.downloader punkt wordnet

# -----------------  START: MODIFICATION -----------------

# Copy the immutable bootstrapper script and make it executable.
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Set the bootstrapper script to run on container start.
ENTRYPOINT ["/app/entrypoint.sh"]
