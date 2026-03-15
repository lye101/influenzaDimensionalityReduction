FROM jupyter/base-notebook:latest

USER root

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install everything as root into the base env explicitly
RUN mamba install -n base -c conda-forge -c pytorch -c nvidia --yes \
    numpy \
    pandas \
    matplotlib \
    scipy \
    scikit-learn \
    tensorflow \
    pytorch \
    seaborn \
    dask \
    distributed \
    dask-jobqueue \
    tensorly \
    polars \
    hdf5 \
    umap-learn \
    biopython \
    toytree \
    ipywidgets \
    && mamba clean --all -f -y

# pip also into the base env explicitly
RUN pip install --no-cache-dir phate tensorly[dask]

USER ${NB_UID}
