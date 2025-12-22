FROM continuumio/miniconda3

# Set the working directory
WORKDIR /mlflow/projects/code

# Copy the MLProject files
COPY conda.yaml .
COPY MLProject .
COPY modelling.py .
COPY preprocessed_data.csv .

# Create the conda environment
RUN conda env create -f conda.yaml

# Activate the environment
SHELL ["conda", "run", "-n", "mlproject", "/bin/bash", "-c"]

# Set the environment path
ENV PATH /opt/conda/envs/mlproject/bin:$PATH

# Command to run the application
CMD ["conda", "run", "--no-capture-output", "-n", "mlproject", "python", "modelling.py"]