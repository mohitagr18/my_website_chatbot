FROM python:3.11-slim

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Expose port
EXPOSE 8080

# Set environment
ENV PORT=8080

# Run Streamlit from deployment folder
CMD ["streamlit", "run", "deployment/streamlit_app.py"]
