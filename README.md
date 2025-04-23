# 🧠 DeepSign — ASL Recognition Backend (CNN API)

This repository contains the backend API for **DeepSign**, a deep learning-powered application for learning American Sign Language (ASL). The backend is responsible for recognizing ASL signs using a custom convolutional neural network (CNN), and serves predictions to the frontend via a FastAPI interface deployed on Google Cloud Platform

This project is part of the Le Wagon's Data Science bootcamp.

## ✨ Features

- Custom CNN Deep Learning model (built from scratch)
- ~1 million parameters, ~20MB size
- Trained on 27 ASL sign classes
- Real-time image input processing using OpenCV
- FastAPI for serving the model as an HTTP API
- Dockerized and deployed via Google Cloud Platform

## 🔗 Dataset

The model is trained on the **27-Class Sign Language Dataset** from Kaggle:  
📊 [Kaggle Dataset Link](https://www.kaggle.com/datasets/ardamavi/27-class-sign-language-dataset/data)

## 🔌 API Integration

This backend is designed to work with the [DeepSign Frontend](https://github.com/your-username/deepsign-frontend).  
It receives frames from the webcam, processes them with OpenCV, and returns predicted ASL signs.

## 🧪 Primary Use Cases

- Recognize ASL signs from real-time webcam input
- Provide feedback for ASL learning applications
- Serve predictions to a frontend client for user interaction

## 🚀 Deployment

The backend is containerized with Docker and deployed on **Google Cloud Platform**. It can easily be adapted to other cloud providers or local environments.

## 🛠️ Technologies Used

- Python  
- TensorFlow / Keras  
- OpenCV  
- FastAPI  
- Docker  
- Google Cloud Platform (GCP)

## ▶️ How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/deepsign-backend.git
   ```

2. Build and run the Docker container:
   ```bash
   docker build -t deepsign .
   docker run -p 8000:8000 deepsign
   ```

3. Access the API locally at:
   ```
   http://localhost:8000/predict
   ```

4. Use the `/docs` endpoint to explore the FastAPI interface:
   ```
   http://localhost:8000/docs
   ```

## 📡 API Endpoint Example

**POST** `/get_image_prediction_prod`  
Payload: Base64 image or raw frame  
Response: 
```python
{
  "filename",
  "prediction",
  "probabilities",
  "model"
}
```
