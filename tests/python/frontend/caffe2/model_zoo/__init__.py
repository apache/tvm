"""Store for caffe2 examples and common models."""
from __future__ import absolute_import as _abs
import os
import sys
import importlib
from . import squeezenet
from caffe2.python.models.download import ModelDownloader

models = [
    'squeezenet',
    'resnet50',
    'vgg19',
]

mf = ModelDownloader()

class Model:
    def __init__(self, model_name):
        self.init_net, self.predict_net, self.value_info = mf.get_c2_model(model_name)

for model in models:
    try:
        locals()['c2_' + model] = importlib.import_module('caffe2.python.models.' + model)
    except ImportError:
        locals()['c2_' + model] = Model(model)

# squeezenet
def relay_squeezenet():
    return squeezenet.get_workload()
