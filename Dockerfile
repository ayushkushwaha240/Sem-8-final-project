# Use an official Python image as the base
FROM python:3.10.12

# Set the working directory inside the container
WORKDIR /app

# Copy the required files to the container
COPY ./requirements.txt /app/requirements.txt
COPY ./main.py /app/main.py

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
