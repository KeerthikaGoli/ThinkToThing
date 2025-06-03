#!/bin/bash

# Create necessary directories
mkdir -p static/images static/models memory

# Start Streamlit UI in the background
streamlit run app_ui.py --server.port 8501 &

# Start the API server
python -m uvicorn main:app --host 0.0.0.0 --port 8888
