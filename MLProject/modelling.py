import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
import cloudpickle
import joblib
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load environment variables
load_dotenv()

# Get DagsHub credentials
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

# Validate credentials
if DAGSHUB_USERNAME and DAGSHUB_TOKEN:
    print("✓ DagsHub credentials loaded successfully!")
    print(f"✓ Username: {DAGSHUB_USERNAME}")
    
    DAGSHUB_REPO_NAME = "Eksperimen_SML_Zayga"
    tracking_uri = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow"
    
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

# Load preprocessed data
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

# Disable autologging
mlflow.sklearn.autolog(disable=True)

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
    
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("test_size", 0.2)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    # Create artifacts directory
    artifacts_dir = "artifacts"
    model_dir = os.path.join(artifacts_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    
    # ========================================
    # SAVE MODEL FILES MANUALLY
    # ========================================
    
    # 1. Save model.pkl with cloudpickle
    model_pkl_path = os.path.join(model_dir, "model.pkl")
    try:
        with open(model_pkl_path, 'wb') as f:
            cloudpickle.dump(model, f)
        print(f"✓ model.pkl saved: {model_pkl_path}")
    except Exception as e:
        print(f"⚠ Cloudpickle failed: {e}, trying joblib...")
        joblib.dump(model, model_pkl_path)
        print(f"✓ model.pkl saved with joblib: {model_pkl_path}")
    
    # 2. Create MLmodel file
    mlmodel_content = f"""artifact_path: model
flavors:
  python_function:
    env:
      conda: conda.yaml
      virtualenv: python_env.yaml
    loader_module: mlflow.sklearn
    model_path: model.pkl
    predict_fn: predict
    python_version: 3.9.25
  sklearn:
    code: null
    pickled_model: model.pkl
    serialization_format: cloudpickle
    sklearn_version: 1.2.2
mlflow_version: 2.9.2
model_size_bytes: {os.path.getsize(model_pkl_path)}
model_uuid: {run.info.run_id}
run_id: {run.info.run_id}
utc_time_created: '{pd.Timestamp.now().isoformat()}'
"""
    
    with open(os.path.join(model_dir, "MLmodel"), 'w') as f:
        f.write(mlmodel_content)
    print(f"✓ MLmodel file created")
    
    # 3. Create conda.yaml
    conda_yaml = """channels:
- conda-forge
dependencies:
- python=3.9.25
- pip<=24.0
- pip:
  - mlflow==2.9.2
  - cloudpickle==3.0.0
  - numpy==1.24.3
  - scikit-learn==1.2.2
name: mlflow-env
"""
    with open(os.path.join(model_dir, "conda.yaml"), 'w') as f:
        f.write(conda_yaml)
    print(f"✓ conda.yaml created")
    
    # 4. Create python_env.yaml
    python_env = """python: 3.9.25
build_dependencies:
- pip<=24.0
dependencies:
- -r requirements.txt
"""
    with open(os.path.join(model_dir, "python_env.yaml"), 'w') as f:
        f.write(python_env)
    print(f"✓ python_env.yaml created")
    
    # 5. Create requirements.txt
    requirements = """mlflow==2.9.2
cloudpickle==3.0.0
numpy==1.24.3
scikit-learn==1.2.2
"""
    with open(os.path.join(model_dir, "requirements.txt"), 'w') as f:
        f.write(requirements)
    print(f"✓ requirements.txt created")
    
    # ========================================
    # LOG MODEL TO MLFLOW TRACKING SERVER
    # ========================================
    try:
        mlflow.sklearn.log_model(model, "model")
        print(f"✓ Model logged to MLflow tracking server")
    except Exception as e:
        print(f"⚠ Failed to log model to tracking server: {e}")
    
    # ========================================
    # VERIFY ALL FILES
    # ========================================
    print("\n" + "=" * 60)
    print("📦 MODEL DIRECTORY CONTENTS:")
    print("=" * 60)
    for root, dirs, files in os.walk(model_dir):
        level = root.replace(model_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            print(f'{subindent}{file} ({file_size:,} bytes)')
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 TRAINING RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("=" * 60)
    print(f"✓ Local artifacts saved to: {os.path.abspath(artifacts_dir)}")
    print(f"✓ MLflow run ID: {run.info.run_id}")
    print("✓ Training completed successfully!")