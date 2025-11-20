FROM jupyter/tensorflow-notebook:python-3.10

RUN pip install --no-cache-dir \
    "numpy<2" \
    tensorly \
    pytensor


USER ${NB_UID}
