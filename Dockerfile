FROM jupyter/tensorflow-notebook:latest

RUN pip install --no-cache-dir \
    numpy<2 \
    tensorly \
    pytensor


USER ${NB_UID}
