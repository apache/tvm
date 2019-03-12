#!/bin/bash

set -e
set -u
set -o pipefail

# fix to certain version for now
pip3 install onnx>=1.1.0

pip3 install https://download.pytorch.org/whl/cu80/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
pip3 install torchvision
