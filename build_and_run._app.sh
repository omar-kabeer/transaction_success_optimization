#!/bin/bash

# Build Docker image
docker build -t transaction-success-predictor:latest .

# Run the container
docker run -d -p 8000:8000 --name transaction-predictor transaction-success-predictor:latest

echo "API is running at http://localhost:8000"
echo "Documentation available at http://localhost:8000/docs"