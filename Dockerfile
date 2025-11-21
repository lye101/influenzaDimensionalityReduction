FROM jupyter/base-notebook:latest

# Switch to root to install packages
USER root

# Update the base system
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Switch back to the notebook user
USER ${NB_UID}

# Install packages using mamba (which is already available in jupyter/base-notebook)
RUN mamba install -c conda-forge -c pytorch -c nvidia --yes \
    numpy \
    pandas \
    matplotlib \
    scipy \
    scikit-learn \
    tensorflow \
    pytorch \
    torchvision \
    torchaudio \
    seaborn \
    dask \
    distributed \
    dask-jobqueue \
    tensorly \
    polars \
    && mamba clean --all -f -y

RUN pip install tensorly[dask]
    
USER ${NB_UID}
