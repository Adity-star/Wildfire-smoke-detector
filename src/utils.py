import os
import shutil
from pathlib import Path
import mlflow 

ROOT_DIR = Path(__file__).resolve().parents[1]  # root directory absolute path

def save_model(experiment_name: str):
    """Save the weights of trained model to the models directory.""" 
    model_weights = os.path.join(ROOT_DIR, "runs", "detect", experiment_name, "weights", "best.pt")
    target_dir = os.path.join(ROOT_DIR, "models")
    os.makedirs(target_dir, exist_ok=True)

    if os.path.exists(model_weights):
        final_model_path = os.path.join(target_dir, "model.pt")
        shutil.copy(src=model_weights, dst=final_model_path)

        # ✅ Log model artifact to MLflow
        mlflow.log_artifact(final_model_path, artifact_path="model")
    else:
        raise FileNotFoundError(f"Model weights not found at: {model_weights}")


def save_metrics_and_params(experiment_name: str) -> None:
    """Save training metrics, params, confusion matrix to reports dir and log to MLflow."""
    path_metrics = os.path.join(ROOT_DIR, "runs", "detect", experiment_name)
    reports_dir = os.path.join(ROOT_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    try:
        # File paths
        files_to_log = {
            "train_metrics.csv": "results.csv",
            "train_confusion_matrix.png": "confusion_matrix.png",
            "train_params.yaml": "args.yaml"
        }

        for dest_name, source_file in files_to_log.items():
            src = os.path.join(path_metrics, source_file)
            dst = os.path.join(reports_dir, dest_name)

            if os.path.exists(src):
                shutil.copy(src, dst)
                mlflow.log_artifact(dst, artifact_path="reports")  # ✅ log each file
            else:
                print(f"⚠️ Skipped logging missing file: {src}")

    except Exception as e:
        raise RuntimeError(f"Failed to save or log artifacts: {e}")
