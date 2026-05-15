# Dockerfile
# Packages the entire app into a portable container image.

# WHY python:3.12-slim: Slim base image removes unnecessary packages,
# keeping the image small (~150MB vs ~900MB for full Python image).
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first — Docker caches this layer.
# WHY copy requirements before code: If only code changes, Docker
# reuses the cached pip install layer and rebuilds faster.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Expose port 8000 for the FastAPI server
EXPOSE 8000

# Start the FastAPI app with uvicorn
# WHY 0.0.0.0: Binds to all network interfaces inside the container
# so traffic from outside the container can reach it.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]