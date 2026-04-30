# HuggingFace Spaces requires port 7860 and user ID 1000
FROM python:3.11-slim

# Create non-root user (HF requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install deps as user
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Pre-download sentence-transformers model at build time
# so cold starts are fast (no download on first request)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy app files
COPY --chown=user . .

# Create data directory
RUN mkdir -p /app/karin_data

EXPOSE 7860

CMD ["gunicorn", "app:app", "--workers", "1", "--timeout", "120", "--bind", "0.0.0.0:7860"]
