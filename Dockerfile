FROM python:3.10-slim

WORKDIR /app

# Upgrade pip and tools for dependency resolution
RUN python -m pip install --upgrade pip setuptools wheel

COPY requirements.txt .

# Let pip resolve subpackages automatically
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    PYTHONUNBUFFERED=1

CMD ["streamlit", "run", "app/main.py"]
