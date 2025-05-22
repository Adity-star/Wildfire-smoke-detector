# Wildfire Smoke Detector 🚒🔥

A machine learning project designed to detect wildfire smoke from images using the state-of-the-art YOLOv8 model. This repository contains everything needed to train, evaluate, and deploy the model for real-time image classification. It integrates with **MLflow** for model tracking and **Gradio** for easy model deployment. Docker is used to ensure consistency across environments and streamline deployment.

---

## 🚀 Features

- **YOLOv8 Model**: A real-time object detection model for wildfire smoke detection.
- **MLflow Integration**: Logs training parameters, evaluation metrics, and model artifacts for reproducibility and easy model versioning.
- **Gradio Interface**: Provides an interactive, user-friendly web interface for real-time predictions.
- **Dockerized**: Ensures a seamless, consistent environment for development, testing, and deployment.

---
## Setup and Installation
### 1. Clone the repository
First, clone the repository to your local machine:

```bash
git clone https://github.com/your-username/Wildfire-smoke-detector.git
cd Wildfire-smoke-detector
```
### 2. Install dependencies
To install the necessary Python packages, run:

```bash
pip install -r requirements.txt
```
Alternatively, if you'd like to use Docker, follow the Docker instructions below.

### 3. Setup DVC and pull data
```bash
dvc remote add -d origin https://dagshub.com/your-username/Wildfire-smoke-detector.dvc
dvc pull
```

---
## Model Training
Prepare your data structure like:
```bash
data/
  ├── train/
  ├── valid/
  ├── test/
  └── data.yaml
```
Then Run:
```bash
dvc repro
```
This command will:
- Run all stages in your pipeline (`train.py`, `evaluate.py`, etc.)
- Track the results and metrics with MLflow
- Store outputs like `model.pt`, `reports/train_metrics.csv`, and more
- Automatically log training progress and results

  ##  Web Interface with Gradio
After training, launch the Gradio app to try your model:
```bash
python app.py
```
Access it at: http://localhost:7860

---

## Using Docker
Docker helps ensure that the project works consistently across different environments. Here's how to set up the project using Docker:

### 1. Build the Docker image
First, build the Docker image from the provided Dockerfile:
```bash
docker build -t wildfire-smoke-detector .
```
### 2. Run the Docker container
After building the image, you can run the application inside a Docker container:
```bash
docker run -p 5000:5000 wildfire-smoke-detector
```
This will start the application, and you can access the Gradio interface at http://localhost:5000.

---
##  [DVC + DAGsHub Data Versioning](https://dagshub.com/Adity-star/Wildfire-smoke-detector)
This project uses DVC (Data Version Control) to manage datasets, model outputs, and ensure reproducibility.

### Set up DVC with DagsHub
1. Initialize DVC (if not done already)
```bash
dvc init
git commit -m "Initialize DVC"
```
2. Add DAGsHub as a DVC remote

```bash
dvc remote add -d origin https://dagshub.com/<username>/<repo>.dvc
git commit .dvc/config -m "Configure DVC remote storage"
```
3. Track your data or model files
```bash
dvc add data/
git add data.dvc .gitignore
git commit -m "Track dataset with DVC"
```
4. Push your data to DagsHub

```bash
dvc push
```
> Note: You must have Git pushed your code first (git push origin main) and be authenticated via HTTPS or SSH.

5. Pull data when cloning repo
```bash
dvc pull
```

## ✍️ Medium Blog

We’ve documented the complete journey of building this wildfire smoke detection system—from data collection to model training, evaluation, and deployment—in a detailed **Medium article**.

📰 **Read the Full Story** → [How We Built a Wildfire Smoke Detection System with YOLOv8 + MLflow + DVC + Gradio](https://medium.com/@your-username/wildfire-smoke-detection-yolov8-mlflow-dvc-docker-gradio-abcdef123456)


> Whether you're a beginner looking to learn or a practitioner aiming to build production-ready pipelines, this blog is for you!

---

##  License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/Adity-star/Wildfire-smoke-detector#) file for details.

## Acknowledgements
- YOLO by [Ultralytics](https://github.com/ultralytics)
- [DAGsHub](https://dagshub.com/dashboard) for data and experiment tracking
- [MLflow](https://mlflow.org/) for model management
- [Gradio](https://www.gradio.app/) for deployment
