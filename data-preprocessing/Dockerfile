FROM python:3.9-slim

WORKDIR /app

# Upgrade pip and install build dependencies
RUN pip install --upgrade pip
RUN apt-get update && \
    apt-get install -y build-essential libatlas-base-dev

# Copy and install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

CMD ["python", "DataProcessing.py"]
