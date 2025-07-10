# Use an official Python runtime as the base image
FROM python:3.9.13-slim

# Set the working directory in the container
WORKDIR /main

# Copy the requirements file into the container
COPY . /api

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt


# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]