# Workflow-CI: MLflow Project dengan CI/CD

Repository ini berisi implementasi CI/CD untuk re-training model machine learning secara otomatis menggunakan MLflow Project dan GitHub Actions.

## 📁 Struktur Repository

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── mlflow-ci.yml          # GitHub Actions workflow
├── .workflow                       # Workflow configuration
├── MLProject/                      # MLflow Project folder
│   ├── modelling.py               # Script training model
│   ├── conda.yaml                 # Environment dependencies
│   ├── MLProject                  # MLflow project configuration
│   ├── preprocessed_data.csv      # Dataset untuk training
│   ├── Dockerfile                 # Docker configuration
│   └── README.md                  # Dokumentasi MLProject
└── README.md                       # Dokumentasi utama (file ini)
```

## 🎯 Fitur

### ✅ Basic (2 pts)
- ✅ Folder MLProject dengan struktur yang sesuai
- ✅ Workflow CI yang dapat melatih model ketika trigger terpantik

### ✅ Skilled (3 pts)
- ✅ Menyimpan artifacts ke GitHub (confusion matrix, feature importance, dll)
- ✅ Artifacts tersimpan selama 30 hari

### ✅ Advance (4 pts)
- ✅ Build dan push Docker Image ke Docker Hub
- ✅ Tagging Docker image dengan commit SHA
- ✅ Link Docker Hub tersimpan sebagai artifact

## 🚀 Cara Setup

### 1. Setup GitHub Repository

1. Buat repository baru di GitHub dengan nama `Workflow-CI`
2. Clone repository ini:
   ```bash
   git clone <your-repo-url>
   cd Workflow-CI
   ```

### 2. Setup GitHub Secrets

Tambahkan secrets berikut di repository Settings → Secrets and variables → Actions:

- `DAGSHUB_USERNAME`: Username DagsHub Anda
- `DAGSHUB_TOKEN`: Token akses DagsHub Anda
- `DOCKER_USERNAME`: Username Docker Hub Anda
- `DOCKER_PASSWORD`: Password atau token Docker Hub Anda

### 3. Konfigurasi DagsHub

1. Buat akun di [DagsHub](https://dagshub.com)
2. Buat repository baru atau gunakan yang sudah ada
3. Dapatkan access token dari Settings → Tokens
4. Update tracking URI di `modelling.py` sesuai repository Anda

### 4. Konfigurasi Docker Hub

1. Buat akun di [Docker Hub](https://hub.docker.com)
2. Buat repository baru (contoh: `mlflow-heart-disease`)
3. Dapatkan access token dari Account Settings → Security

## 🔄 Cara Kerja Workflow

### Trigger Workflow

Workflow akan berjalan otomatis ketika:
- Push ke branch `main` atau `master` dengan perubahan di folder `MLProject/`
- Pull request ke branch `main` atau `master`
- Manual trigger dari GitHub Actions tab

### Job 1: Train Model

1. Checkout kode dari repository
2. Setup Python 3.8 dan Miniconda
3. Install MLflow
4. Setup kredensial DagsHub dari secrets
5. Jalankan MLflow project: `mlflow run . --env-manager=conda`
6. Upload artifacts (confusion matrix, feature importance, ROC curve, dll) ke GitHub

### Job 2: Build Docker

1. Checkout kode dari repository
2. Setup Docker Buildx
3. Login ke Docker Hub menggunakan secrets
4. Build Docker image dari Dockerfile
5. Push image dengan 2 tags:
   - `latest`: Tag untuk versi terbaru
   - `<commit-sha>`: Tag dengan commit hash untuk versioning
6. Simpan link Docker Hub sebagai artifact

## 📊 Hasil Training

Setiap kali workflow berjalan, hasil berikut akan tersimpan:

### Di DagsHub:
- Model yang sudah dilatih
- Metrics (accuracy, precision, recall, F1-score, ROC AUC)
- Parameters yang digunakan
- Semua artifacts

### Di GitHub Actions:
- Confusion Matrix (PNG)
- Feature Importance (PNG & CSV)
- ROC Curve (PNG)
- Classification Report (CSV)
- Model Summary (CSV)

### Di Docker Hub:
- Docker image yang siap dijalankan
- Multiple tags untuk version control

## 🐳 Menggunakan Docker Image

Untuk menjalankan model menggunakan Docker:

```bash
# Pull image
docker pull <your-dockerhub-username>/mlflow-heart-disease:latest

# Jalankan container
docker run -e DAGSHUB_USERNAME=<your-username> \
           -e DAGSHUB_TOKEN=<your-token> \
           <your-dockerhub-username>/mlflow-heart-disease:latest
```

## 🔧 Menjalankan Secara Lokal

### Dengan MLflow:
```bash
cd MLProject
mlflow run . --env-manager=conda
```

### Dengan Python langsung:
```bash
cd MLProject
conda env create -f conda.yaml
conda activate mlproject
python modelling.py
```

### Dengan Docker:
```bash
cd MLProject
docker build -t mlflow-heart-disease .
docker run -e DAGSHUB_USERNAME=<your-username> \
           -e DAGSHUB_TOKEN=<your-token> \
           mlflow-heart-disease
```

## 📋 Requirements

- Python 3.8
- MLflow
- Conda/Miniconda
- Docker (untuk build image)
- Akun DagsHub (untuk tracking experiments)
- Akun Docker Hub (untuk menyimpan images)

## 🔍 Monitoring

### GitHub Actions
1. Buka tab **Actions** di repository GitHub
2. Pilih workflow run yang ingin dilihat
3. Klik job untuk melihat logs detail
4. Download artifacts dari bagian bawah summary

### DagsHub
1. Buka repository DagsHub Anda
2. Klik tab **Experiments**
3. Lihat metrics, parameters, dan artifacts dari setiap run

### Docker Hub
1. Buka repository Docker Hub Anda
2. Lihat semua tags yang tersedia
3. Pull image yang diinginkan

## 🎓 Penilaian Kriteria

Implementasi ini mencapai level **Advance (4 pts)** dengan fitur:

✅ Membuat folder MLProject dengan struktur yang benar
✅ Membuat workflow CI yang dapat melatih model ketika trigger terpantik
✅ Menyimpan artifacts ke GitHub repository
✅ Build dan push Docker image ke Docker Hub
✅ Menggunakan MLflow Project untuk manajemen workflow

## 🤝 Kontribusi

Untuk berkontribusi:
1. Fork repository ini
2. Buat branch baru: `git checkout -b feature/nama-fitur`
3. Commit changes: `git commit -m 'Menambahkan fitur'`
4. Push ke branch: `git push origin feature/nama-fitur`
5. Buat Pull Request

## 📝 Catatan

- Pastikan semua secrets sudah dikonfigurasi dengan benar di GitHub
- Dataset `preprocessed_data.csv` harus sudah siap dan dalam format yang benar
- Tracking URI DagsHub di `modelling.py` harus disesuaikan dengan repository Anda
- Nama Docker image di workflow harus sesuai dengan repository Docker Hub Anda

## 📄 Lisensi

MIT License - Silakan gunakan dan modifikasi sesuai kebutuhan.

---

**Dibuat dengan ❤️ untuk pembelajaran Machine Learning Operations (MLOps)**