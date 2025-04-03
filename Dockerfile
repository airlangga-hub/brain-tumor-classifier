FROM python:3.10-slim

WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Upgrade pip, then install other dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
