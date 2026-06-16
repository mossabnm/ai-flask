FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Download the real model from Git LFS directly since Railway doesn't pull LFS files automatically
RUN python -c "import urllib.request; urllib.request.urlretrieve('https://media.githubusercontent.com/media/mossabnm/ai-flask/master/best_model.keras', 'best_model.keras')"

# Set port
EXPOSE 8080

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV MODEL_PATH=best_model.keras

# Run gunicorn with dynamic PORT
CMD sh -c "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --timeout 120 app:app"
