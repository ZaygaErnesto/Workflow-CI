name: MLflow CI/CD Pipeline

on:
  push:
    branches: [ main, master ]
    paths:
      - 'MLProject/**'
  pull_request:
    branches: [ main, master ]
  workflow_dispatch:

jobs:
  train-model:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout repository
      uses: actions/checkout@v3
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.8'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install mlflow
        pip install conda
        
    - name: Install Miniconda
      uses: conda-incubator/setup-miniconda@v2
      with:
        auto-update-conda: true
        python-version: 3.8
        
    - name: Set up DagsHub credentials
      env:
        DAGSHUB_USERNAME: ${{ secrets.DAGSHUB_USERNAME }}
        DAGSHUB_TOKEN: ${{ secrets.DAGSHUB_TOKEN }}
      run: |
        echo "DAGSHUB_USERNAME=$DAGSHUB_USERNAME" >> $GITHUB_ENV
        echo "DAGSHUB_TOKEN=$DAGSHUB_TOKEN" >> $GITHUB_ENV
        
    - name: Run MLflow Project
      env:
        DAGSHUB_USERNAME: ${{ secrets.DAGSHUB_USERNAME }}
        DAGSHUB_TOKEN: ${{ secrets.DAGSHUB_TOKEN }}
      run: |
        cd MLProject
        mlflow run . --no-conda
        
    - name: Upload artifacts to GitHub
      uses: actions/upload-artifact@v3
      with:
        name: model-artifacts
        path: |
          MLProject/*.png
          MLProject/*.csv
          MLProject/mlruns/
        retention-days: 30
        
  build-docker:
    needs: train-model
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout repository
      uses: actions/checkout@v3
      
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
      
    - name: Login to Docker Hub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
        
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: ./MLProject
        file: ./MLProject/Dockerfile
        push: true
        tags: ${{ secrets.DOCKER_USERNAME }}/mlflow-heart-disease:latest,${{ secrets.DOCKER_USERNAME }}/mlflow-heart-disease:${{ github.sha }}
        
    - name: Create Docker Hub link file
      run: |
        echo "Docker Image: https://hub.docker.com/r/${{ secrets.DOCKER_USERNAME }}/mlflow-heart-disease" > MLProject/docker-hub-link.txt
        echo "Latest Tag: ${{ secrets.DOCKER_USERNAME }}/mlflow-heart-disease:latest" >> MLProject/docker-hub-link.txt
        echo "Commit Tag: ${{ secrets.DOCKER_USERNAME }}/mlflow-heart-disease:${{ github.sha }}" >> MLProject/docker-hub-link.txt
        
    - name: Upload Docker Hub link
      uses: actions/upload-artifact@v3
      with:
        name: docker-info
        path: MLProject/docker-hub-link.txt
        retention-days: 90