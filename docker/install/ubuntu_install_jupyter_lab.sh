#!/bin/bash

set -e
set -u
set -o pipefail

# Install Jupyter Lab
pip3 install notebook jupyterhub jupyterlab

# Install extensions
pip3 install ipywidgets

# Enable extensions
jupyter nbextension enable --py widgetsnbextension

# Install jupyter-server-proxy
pip3 install jupyter-server-proxy
jupyter labextension install @jupyterlab/server-proxy