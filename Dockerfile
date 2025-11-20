FROM jupyter/tensorflow-notebook:latest

# Add lightweight tensor packages
RUN pip install --no-cache-dir \
    tensorly \
    tensortools \
    scikit-tensor \
    pytensor

USER ${NB_UID}
