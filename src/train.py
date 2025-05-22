import os
from pathlib import Path
import yaml
import mlflow
from dotenv import load_dotenv
from ultralytics import YOLO
from utils import save_model, save_metrics_and_params
import torch

# Load environment variables
load_dotenv()
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')

# Define paths
root_dir = Path(__file__).resolve().parents[1]
params_path = root_dir / "params.yaml"
data_yaml_path = root_dir / "data" / "data.yaml"
metrics_path = root_dir / "reports/train_metrics.json"

def load_params(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def train_yolo_model(params, data_yaml_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Use CUDA if available

    # Load the model (use pre-trained weights if needed)
    model = YOLO(params['model_type'])

    # Ensure the model uses the GPU if available
    model.to(device)

    # Train the model
    results = model.train(
        data=str(data_yaml_path),
        imgsz=params['imgsz'],
        batch=params['batch'],
        epochs=params['epochs'],
        optimizer=params['optimizer'],
        lr0=params['lr0'],
        seed=params['seed'],
        pretrained=params['pretrained'],
        name=params['name'],
        device=device,  # Set the device here (0 for GPU if available)
        workers=8,  # Increase the number of workers for data loading
        amp=True,  # Automatic mixed precision (speed-up with little loss in precision)
        cache=True,  # Cache images for faster loading
        cos_lr=True,  # Cosine learning rate schedule
        patience=3,  # Early stopping patience
        freeze=[0]  # Freeze initial layers (if you are fine-tuning)
    )

    return model, results

def evaluate_model(model):
    # Evaluate the model performance
    metrics = model.val()
    mlflow.log_metric("mAP50", metrics.box.map50)
    mlflow.log_metric("mAP50-95", metrics.box.map)

def log_params(params):
    mlflow.log_params({
        'model_type': params['model_type'],
        'epochs': params['epochs'],
        'optimizer': params['optimizer'],
        'learning_rate': params['lr0'],
        'imgsz': params['imgsz'],
        'batch': params['batch']
    })

if __name__ == '__main__':
    # Load parameters
    params = load_params(params_path)

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Start an MLflow run
    with mlflow.start_run(run_name=params['name']):
        print("Starting training with YOLOv8...")
        
        # Train the YOLO model
        model, results = train_yolo_model(params, data_yaml_path)

        # Log the training parameters
        print("Logging training parameters...")
        log_params(params)

        # Evaluate the model after training
        print("Evaluating model...")
        evaluate_model(model)

        # Save the trained model and metrics
        print("Saving model and metrics...")
        save_model(experiment_name=params['name'])
        save_metrics_and_params(experiment_name=params['name'])

        print("Training pipeline completed.")
