# Agentic Architecture Mediation System
# Multi-agent system for LLM-assisted architecture governance
# Accompanies IEEE Software 2025 paper

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config/ ./config/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY main.py .
COPY README.md .
COPY LICENSE .

# Create outputs directory
RUN mkdir -p outputs/figures

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OPENAI_API_KEY=""

# Default command runs demo scenarios
CMD ["python", "scripts/demo_cogent.py"]

# Alternative commands:
# Run interactive mode: docker run -it --entrypoint python agentsysarch main.py
# Run visualizations: docker run --entrypoint python agentsysarch scripts/visualize_results.py
