# Use a specific Python version to avoid mismatches
FROM python:3.12.4-bookworm

# Environment variables
ENV PYTHONUNBUFFERED True
ENV APP_HOME=/back-end

# Set the working directory
WORKDIR $APP_HOME

# Install PortAudio dependencies
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
# Copy the entire project to the container's working directory
COPY . ./

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get install ffmpeg

# Specify the entry point for the container
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app

