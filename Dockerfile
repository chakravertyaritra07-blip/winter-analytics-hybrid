# Use a lightweight Python Linux image
FROM python:3.10-slim

# Prevent Python from writing .pyc files & buffer stdout (better logs)
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies (needed for Python math libs)
# We do NOT install R here
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Run the server
# Make sure to replace 'winter_project' with your actual project folder name if different
CMD ["gunicorn", "winter_project.wsgi:application", "--bind", "0.0.0.0:8000"]