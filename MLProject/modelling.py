import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env dari root folder (parent directory) atau dari environment variables
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"✓ Loaded .env from: {env_path}")

# Get credentials from environment variables (bisa dari .env atau GitHub Secrets)
dagshub_username = os.getenv('DAGSHUB_USERNAME')
dagshub_token = os.getenv('DAGSHUB_TOKEN')

if not dagshub_username or not dagshub_token:
    raise ValueError(
        "DAGSHUB_USERNAME atau DAGSHUB_TOKEN tidak ditemukan!\n"
        "Pastikan environment variables sudah di-set (melalui .env atau GitHub Secrets)\n"
        f"DAGSHUB_USERNAME: {dagshub_username}\n"
        f"DAGSHUB_TOKEN: {'Set' if dagshub_token else 'Not Set'}"
    )

# Setup DagsHub
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/ZaygaErnesto/Eksperimen_SML_Zayga.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_username
os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

print("✓ DagsHub credentials loaded successfully!")
print(f"✓ Tracking URI: {os.environ['MLFLOW_TRACKING_URI']}")
print(f"✓ Username: {dagshub_username}")

# Load data
data_path = os.getenv('DATA_PATH', './preprocessed_data.csv')
df = pd.read_csv(data_path)

# Separate features and target
X = df.drop('Target', axis=1)
y = df['Target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set MLflow experiment
mlflow.set_experiment("Advance_DagsHub_Training")

# Disable autologging
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

with mlflow.start_run(run_name="RandomForest_DagsHub_Advanced"):
    # Train with hyperparameter tuning
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    
    # Predictions
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)
    
    # Calculate metrics (autolog metrics)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Manual logging - Parameters
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    
    # Manual logging - Metrics (autolog)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("training_score", grid_search.best_score_)
    
    # Additional metrics (not in autolog)
    if len(np.unique(y)) == 2:  # Binary classification
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        mlflow.log_metric("roc_auc", roc_auc)
    
    # Log model
    mlflow.sklearn.log_model(best_model, "model")
    
    # ARTIFACT 1: Confusion Matrix (Additional)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')
    plt.close()
    
    # ARTIFACT 2: Feature Importance (Additional)
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
    plt.savefig('feature_importance.png')
    mlflow.log_artifact('feature_importance.png')
    plt.close()
    
    # Save feature importance as CSV
    feature_importance.to_csv('feature_importance.csv', index=False)
    mlflow.log_artifact('feature_importance.csv')
    
    # ARTIFACT 3: Classification Report (Additional)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv('classification_report.csv')
    mlflow.log_artifact('classification_report.csv')
    
    # ARTIFACT 4: ROC Curve (Additional - if binary classification)
    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve.png')
        mlflow.log_artifact('roc_curve.png')
        plt.close()
    
    # ARTIFACT 5: Model Performance Summary (Additional)
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
    summary_df.to_csv('model_summary.csv', index=False)
    mlflow.log_artifact('model_summary.csv')
    
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
    print(f"Check results at: {os.environ['MLFLOW_TRACKING_URI']}")

print("Advanced model training completed!")