"""Store for caffe2 examples and common models."""
from __future__ import absolute_import as _abs
import os
import importlib

models = [
    'squeezenet',
    'resnet50',
    'vgg19',
]

# skip download if model exist
for model in models:
    try:
        locals()['c2_' + model] = importlib.import_module('caffe2.python.models.' + model)
    except ImportError:
        os.system("python -m caffe2.python.models.download -i -f " + model)
        locals()['c2_' + model] = importlib.import_module('caffe2.python.models.' + model)
