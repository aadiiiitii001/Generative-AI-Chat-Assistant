# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Upgrade pip (prevents dependency resolution issues)
RUN python -m pip install --upgrade pip

# Copy dependency file first (for Docker layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Set Streamlit environment variables (optional but recommended)
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    PYTHONUNBUFFERED=1

# Command to run Streamlit app
CMD ["streamlit", "run", "app/main.py"]
