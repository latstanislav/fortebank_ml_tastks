from pathlib import Path

# Пути
ROOT_DIR = Path(__file__).parent.parent  # корень проекта
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"

# MLflow
MLFLOW_TRACKING_URI = "http://localhost:5000"
