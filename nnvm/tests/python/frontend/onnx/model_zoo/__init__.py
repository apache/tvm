"""Store for onnx examples and common models."""
from __future__ import absolute_import as _abs
import os
import logging
from .super_resolution import get_super_resolution

def _download(url, filename, overwrite=False):
    if os.path.isfile(filename) and not overwrite:
        logging.debug('File %s existed, skip.', filename)
        return
    logging.debug('Downloading from url %s to %s', url, filename)
    try:
        import urllib.request
        urllib.request.urlretrieve(url, filename)
    except:
        import urllib
        urllib.urlretrieve(url, filename)

def _as_abs_path(fname):
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(cur_dir, fname)


URLS = {
    'super_resolution.onnx': 'https://gist.github.com/zhreshold/bcda4716699ac97ea44f791c24310193/raw/93672b029103648953c4e5ad3ac3aadf346a4cdc/super_resolution_0.2.onnx',
    'squeezenet1_1.onnx': 'https://gist.github.com/zhreshold/bcda4716699ac97ea44f791c24310193/raw/93672b029103648953c4e5ad3ac3aadf346a4cdc/squeezenet1_1_0.2.onnx',
    'lenet.onnx': 'https://gist.github.com/zhreshold/bcda4716699ac97ea44f791c24310193/raw/93672b029103648953c4e5ad3ac3aadf346a4cdc/lenet_0.2.onnx',
    'resnet18_1_0.onnx': 'https://gist.github.com/zhreshold/bcda4716699ac97ea44f791c24310193/raw/b385b1b242dc89a35dd808235b885ed8a19aedc1/resnet18_1.0.onnx'}

# download and add paths
for k, v  in URLS.items():
    name = k.split('.')[0]
    path = _as_abs_path(k)
    _download(v, path, False)
    locals()[name] = path

# symbol for graph comparison
super_resolution_sym = get_super_resolution()
