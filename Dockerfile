# Dockerfile
# Packages the entire app into a portable container image.

# WHY python:3.12-slim: Slim base image removes unnecessary packages,
# keeping the image small (~150MB vs ~900MB for full Python image).
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY train_model.py .

# Train and save model at build time
# WHY: This ensures the model is always compatible with the
# installed sklearn version — no pickle version mismatch possible.
RUN python train_model.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]