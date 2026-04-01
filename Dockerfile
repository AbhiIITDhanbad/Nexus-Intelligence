# Use a lightweight Python base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project structure into the container
COPY . .

# Set PYTHONPATH so your sys.path.append() logic in app.py works flawlessly
ENV PYTHONPATH="/app/Advanced_RAG/src:/app"