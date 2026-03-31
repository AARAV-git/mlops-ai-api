# 🚀 MLOps AI API Deployment

## 🔹 Overview
This project demonstrates end-to-end deployment of a FastAPI-based AI API using:

- Docker (containerization)
- Azure Container Registry (ACR)
- Azure Kubernetes Service (AKS)

## 🔹 Pipeline
FastAPI → Docker → ACR → AKS → LoadBalancer → Public API

## 🔹 Tech Stack
- FastAPI
- Docker
- Kubernetes (AKS)
- Azure Cloud

## 🔹 Run Locally
```bash
docker build -t ai-api .
docker run -p 8000:8000 ai-api
