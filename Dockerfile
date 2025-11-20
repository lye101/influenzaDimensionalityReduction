FROM jupyter/tensorflow-notebook:latest

# Add only compatible tensor packages
RUN pip install --no-cache-dir \
    tensorly \
    pytensor

# Note: tensortools and scikit-tensor have compatibility issues
# They require older NumPy/Python versions

USER ${NB_UID}
