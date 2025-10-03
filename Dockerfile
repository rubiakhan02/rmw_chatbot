# Step 1: Use an official Python 3.13 slim image
FROM python:3.13-slim

# Step 2: Set working directory
WORKDIR /app

# Step 3: Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Step 4: Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Step 5: Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Step 6: Upgrade pip
RUN pip install --upgrade pip

# Step 7: Install heavy dependencies first
RUN pip install numpy==1.26.4 sentence-transformers==2.2.2 faiss-cpu==1.12.0 --no-cache-dir

# Step 8: Install remaining dependencies
RUN pip install -r requirements.txt --no-cache-dir

# Step 9: Copy project files
COPY . .

# Step 10: Expose port for Render (default 10000 for web services)
EXPOSE 10000

# Step 11: Set environment variable for Flask/Gradio to run on all interfaces
ENV HOST=0.0.0.0 PORT=10000

# Step 12: Run the chatbot
CMD ["python", "app.py"]
