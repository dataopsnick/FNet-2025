ARG REGION=us-west-2
FROM 763104351884.dkr.ecr.${REGION}.amazonaws.com/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-sagemaker

# Install git for cloning the repository
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install base requirements. These will be cached by Docker,
# speeding up builds unless this file changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m nltk.downloader punkt wordnet

# The source code will now be cloned at runtime inside the container.
# We no longer need to copy the src directory into the image.
# REMOVED: COPY src/ /app/src
# REMOVED: COPY src/ds_config.json /app/src/ds_config.json

# Copy the new entrypoint script and make it executable
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Set the new entrypoint script to run on container start
ENTRYPOINT ["/app/entrypoint.sh"]