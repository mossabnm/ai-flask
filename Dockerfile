FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY best_model.keras .

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV MODEL_PATH=best_model.keras

# Run gunicorn with dynamic PORT
CMD sh -c "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --timeout 120 app:app"
