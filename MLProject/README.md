# MLProject: Heart Disease Model Training

Folder ini berisi implementasi MLflow Project untuk training model prediksi penyakit jantung dengan RandomForestClassifier.

## 📁 File Structure

```
MLProject/
├── modelling.py              # Script utama untuk training model
├── conda.yaml                # Definisi environment conda
├── MLProject                 # Konfigurasi MLflow Project
├── preprocessed_data.csv     # Dataset yang sudah dipreprocess
├── Dockerfile                # Docker configuration untuk deployment
└── README.md                 # Dokumentasi ini
```

## 🎯 Fitur Model

### Model: RandomForestClassifier
- **Hyperparameter Tuning**: GridSearchCV dengan cross-validation 5-fold
- **Metrics yang diukur**:
  - Accuracy
  - Precision (weighted)
  - Recall (weighted)
  - F1-Score (weighted)
  - ROC AUC (untuk binary classification)

### Artifacts yang Dihasilkan

1. **confusion_matrix.png**: Visualisasi confusion matrix
2. **feature_importance.png**: Top 15 fitur terpenting
3. **feature_importance.csv**: Daftar lengkap feature importance
4. **classification_report.csv**: Laporan klasifikasi lengkap
5. **roc_curve.png**: Kurva ROC (jika binary classification)
6. **model_summary.csv**: Ringkasan performa model

## 🚀 Cara Menggunakan

### 1. Dengan MLflow Project (Recommended)

```bash
# Dari folder MLProject
mlflow run . --env-manager=conda

# Dengan parameter custom
mlflow run . --env-manager=conda -P test_size=0.3 -P random_state=123
```

### 2. Dengan Python Langsung

```bash
# Create dan aktivasi environment
conda env create -f conda.yaml
conda activate mlproject

# Jalankan training
python modelling.py
```

### 3. Dengan Docker

```bash
# Build image
docker build -t heart-disease-model .

# Run container
docker run -e DAGSHUB_USERNAME=<your-username> \
           -e DAGSHUB_TOKEN=<your-token> \
           heart-disease-model
```

## ⚙️ Konfigurasi

### Environment Variables

Script memerlukan environment variables berikut:

- `DAGSHUB_USERNAME`: Username DagsHub untuk MLflow tracking
- `DAGSHUB_TOKEN`: Token akses DagsHub
- `DATA_PATH` (optional): Path ke dataset, default: `./preprocessed_data.csv`

### Parameter MLflow Project

Parameter yang dapat di-customize di MLProject:

- `test_size` (float, default: 0.2): Proporsi data untuk testing
- `random_state` (int, default: 42): Random seed untuk reproducibility
- `n_estimators` (str, default: "100,200,300"): Jumlah trees untuk grid search
- `max_depth` (str, default: "15,20,None"): Max depth untuk grid search
- `min_samples_split` (str, default: "2,5"): Min samples split untuk grid search
- `min_samples_leaf` (str, default: "1,2"): Min samples leaf untuk grid search

## 📊 Dataset Requirements

Dataset `preprocessed_data.csv` harus memiliki format:
- Kolom terakhir bernama `Target`: Label kelas
- Kolom lainnya: Features yang sudah dinormalisasi/standardisasi
- Tidak ada missing values
- Semua kolom numeric

## 📈 Tracking dengan DagsHub

Hasil training akan otomatis dicatat ke DagsHub MLflow:

1. Login ke [DagsHub](https://dagshub.com)
2. Buka repository Anda
3. Klik tab **Experiments**
4. Lihat semua runs, metrics, parameters, dan artifacts

---

**Happy Training! 🚀**

To run the Docker container:
```bash
docker run <image-name>
```

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License - see the LICENSE file for details.