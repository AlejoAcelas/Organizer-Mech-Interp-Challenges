FROM pytorch/pytorch:latest

# Set working directory
WORKDIR /app

# Install additional dependencies
RUN pip install einops jaxtyping numpy transformer_lens
