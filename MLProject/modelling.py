import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load environment variables
load_dotenv()

# Get DagsHub credentials
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

# Validate credentials (optional for local run)
if DAGSHUB_USERNAME and DAGSHUB_TOKEN:
    print("✓ DagsHub credentials loaded successfully!")
    print(f"✓ Username: {DAGSHUB_USERNAME}")
    
    # Configure MLflow with DagsHub
    DAGSHUB_REPO_NAME = "Eksperimen_SML_Zayga"
    tracking_uri = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow"
    
    # Clear existing run ID
    if 'MLFLOW_RUN_ID' in os.environ:
        print(f"⚠ Clearing existing MLFLOW_RUN_ID: {os.environ['MLFLOW_RUN_ID']}")
        del os.environ['MLFLOW_RUN_ID']
    
    os.environ['MLFLOW_TRACKING_URI'] = tracking_uri
    os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN
    
    mlflow.set_tracking_uri(tracking_uri)
    print(f"✓ Tracking URI: {tracking_uri}")
else:
    print("⚠ Running without DagsHub tracking (local mode)")
    mlflow.set_tracking_uri("file:./mlruns")

# Load preprocessed data - Multiple path fallbacks
data_paths = [
    os.path.join(os.path.dirname(__file__), 'preprocessed_data.csv'),
    'preprocessed_data.csv',
    '../preprocessing/processed_data.csv',
    os.path.join(os.path.dirname(__file__), '..', 'preprocessing', 'processed_data.csv')
]

data_path = None
for path in data_paths:
    if os.path.exists(path):
        data_path = path
        break

if data_path is None:
    raise FileNotFoundError(
        f"Data file not found! Tried paths:\n" + 
        "\n".join(f"  - {p}" for p in data_paths)
    )

df = pd.read_csv(data_path)
print(f"✓ Data loaded from: {data_path}")
print(f"✓ Shape: {df.shape[0]} rows, {df.shape[1]} columns")

# Prepare features and target
X = df.drop('Target', axis=1)
y = df['Target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set MLflow experiment
mlflow.set_experiment("Basic_Model_Training")

# Disable autologging to avoid conflicts with mlflow run
mlflow.sklearn.autolog(disable=True)

# Start MLflow run
print("✓ Starting MLflow run...")

with mlflow.start_run(run_name="RandomForest_Basic_Model") as run:
    print(f"✓ Run ID: {run.info.run_id}")
    
    # Train model
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Manual logging - Parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("test_size", 0.2)
    
    # Manual logging - Metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    # Create artifacts directory
    artifacts_dir = "artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # ========================================
    # IMPORTANT: Save model locally for Docker build
    # ========================================
    model_dir = os.path.join(artifacts_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    
    # Save using mlflow.sklearn.save_model (creates MLmodel file)
    mlflow.sklearn.save_model(model, model_dir)
    print(f"✓ Model saved locally to: {model_dir}")
    
    # Also log model to MLflow tracking server
    mlflow.sklearn.log_model(model, "model")
    print(f"✓ Model logged to MLflow tracking server")
    
    # Print results
    print("=" * 50)
    print("BASIC MODEL TRAINING RESULTS")
    print("=" * 50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("=" * 50)
    print(f"✓ Local artifacts saved to: {os.path.abspath(artifacts_dir)}")
    print("✓ Training completed successfully!")