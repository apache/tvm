"""Store for caffe2 examples and common models."""
from __future__ import absolute_import as _abs
import os
import sys
import importlib
from . import squeezenet

models = [
    'squeezenet',
    'resnet50',
    'vgg19',
]

base_url = "https://s3.amazonaws.com/download.caffe2.ai/models"
# save the model data temporary for the test
model_base_dir = "/tmp"

def _download(model, overwrite=False):
    model_dir = '{folder}/{m}'.format(folder=model_base_dir, m=model)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    for filename in ['predict_net.pb', 'init_net.pb']:
        if os.path.isfile('{folder}/{f}'.format(folder=model_dir, f=filename)) and not overwrite:
            return model_dir
        try:
            import urllib.request
            urllib.request.urlretrieve('{url}/{m}/{f}'.format(url=base_url, m=model, f=filename),
                                       '{folder}/{f}'.format(folder=model_dir, f=filename))
        except Exception:
            import urllib
            urllib.urlretrieve('{url}/{m}/{f}'.format(url=base_url, m=model, f=filename),
                               '{folder}/{f}'.format(folder=model_dir, f=filename))

    os.symlink("{folder}/__sym_init__.py".format(folder=os.path.abspath(os.path.dirname(__file__))),
               "{folder}/__init__.py".format(folder=model_dir))
    return model_dir



def _as_abs_path(fname):
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(cur_dir, fname)


# skip download if model exist
for model in models:
    try:
        raise ImportError
        locals()['c2_' + model] = importlib.import_module('caffe2.python.models.' + model)
    except ImportError:
        try:
            raise ModuleNotFoundError
            os.system("python3 -m caffe2.python.models.download -i -f " + model)
            locals()['c2_' + model] = importlib.import_module('caffe2.python.models.' + model)
        except ModuleNotFoundError:
            _download(model)
            if model_base_dir not in sys.path:
                sys.path.append(model_base_dir)
            locals()['c2_' + model] = importlib.import_module(model)

# squeezenet
def relay_squeezenet():
    return squeezenet.get_workload()
