import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            roc_auc_score, roc_curve, auc)

# Load environment variables
load_dotenv()

# Get DagsHub credentials
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

# Validate credentials
if not DAGSHUB_USERNAME or not DAGSHUB_TOKEN:
    raise ValueError(
        "DAGSHUB_USERNAME atau DAGSHUB_TOKEN tidak ditemukan!\n"
        "Pastikan environment variables sudah di-set (melalui .env atau GitHub Secrets)\n"
        f"DAGSHUB_USERNAME: {DAGSHUB_USERNAME or 'Not Set'}\n"
        f"DAGSHUB_TOKEN: {'Set' if DAGSHUB_TOKEN else 'Not Set'}"
    )

print("✓ DagsHub credentials loaded successfully!")
print(f"✓ Username: {DAGSHUB_USERNAME}")

# Configure MLflow with DagsHub
DAGSHUB_REPO_NAME = "Eksperimen_SML_Zayga"  # Ganti dengan nama repo Anda
tracking_uri = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow"

# IMPORTANT: Clear any existing run ID from environment to avoid conflicts
# This happens when running via 'mlflow run' CLI which sets MLFLOW_RUN_ID
if 'MLFLOW_RUN_ID' in os.environ:
    print(f"⚠ Clearing existing MLFLOW_RUN_ID: {os.environ['MLFLOW_RUN_ID']}")
    del os.environ['MLFLOW_RUN_ID']

os.environ['MLFLOW_TRACKING_URI'] = tracking_uri
os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN

mlflow.set_tracking_uri(tracking_uri)
print(f"✓ Tracking URI: {tracking_uri}")

# Load preprocessed data
data_path = os.path.join(os.path.dirname(__file__), 'preprocessed_data.csv')
if not os.path.exists(data_path):
    data_path = 'preprocessed_data.csv'

df = pd.read_csv(data_path)
print(f"✓ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Prepare features and target
X = df.drop('Target', axis=1)
y = df['Target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set MLflow experiment
mlflow.set_experiment("Advance_DagsHub_Training")

# Disable autologging to avoid conflicts
mlflow.sklearn.autolog(disable=True)

# Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [15, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Perform Grid Search
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Always create a fresh run - we cleared MLFLOW_RUN_ID above
print("✓ Creating new MLflow run...")

with mlflow.start_run(run_name="RandomForest_DagsHub_Advanced") as run:
    print(f"✓ Run started with ID: {run.info.run_id}")
    
    # Train with hyperparameter tuning
    print("Training model with GridSearchCV...")
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    
    # Predictions
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Manual logging - Parameters
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    
    # Manual logging - Metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("training_score", grid_search.best_score_)
    
    # Additional metrics for binary classification
    if len(np.unique(y)) == 2:
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        mlflow.log_metric("roc_auc", roc_auc)
    
    # Log model
    mlflow.sklearn.log_model(best_model, "model")
    
    # Create artifacts directory
    artifacts_dir = "artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # ARTIFACT 1: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    cm_path = os.path.join(artifacts_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    plt.close()
    
    # ARTIFACT 2: Feature Importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance['feature'][:15], feature_importance['importance'][:15])
    plt.xlabel('Importance')
    plt.title('Top 15 Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    fi_path = os.path.join(artifacts_dir, 'feature_importance.png')
    plt.savefig(fi_path)
    mlflow.log_artifact(fi_path)
    plt.close()
    
    # Save feature importance as CSV
    fi_csv_path = os.path.join(artifacts_dir, 'feature_importance.csv')
    feature_importance.to_csv(fi_csv_path, index=False)
    mlflow.log_artifact(fi_csv_path)
    
    # ARTIFACT 3: Classification Report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_path = os.path.join(artifacts_dir, 'classification_report.csv')
    report_df.to_csv(report_path)
    mlflow.log_artifact(report_path)
    
    # ARTIFACT 4: ROC Curve (if binary classification)
    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc_val = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc_val:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        roc_path = os.path.join(artifacts_dir, 'roc_curve.png')
        plt.savefig(roc_path)
        mlflow.log_artifact(roc_path)
        plt.close()
    
    # ARTIFACT 5: Model Performance Summary
    summary = {
        'Model': 'RandomForestClassifier',
        'Best Parameters': str(grid_search.best_params_),
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Training Score': grid_search.best_score_
    }
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(artifacts_dir, 'model_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    mlflow.log_artifact(summary_path)
    
    print("=" * 50)
    print("ADVANCED MODEL TRAINING RESULTS (DagsHub)")
    print("=" * 50)
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    if len(np.unique(y)) == 2:
        print(f"ROC AUC: {roc_auc:.4f}")
    print("=" * 50)
    print(f"✓ Results logged to: {tracking_uri}")
    print(f"✓ Run ID: {run.info.run_id}")
    print("✓ Training completed successfully!")