"""Store for onnx examples and common models."""
from __future__ import absolute_import as _abs
import os
import logging
from .super_resolution import get_super_resolution
from tvm.contrib.download import download_testdata


URLS = {
    'super_resolution.onnx': 'https://gist.github.com/zhreshold/bcda4716699ac97ea44f791c24310193/raw/93672b029103648953c4e5ad3ac3aadf346a4cdc/super_resolution_0.2.onnx',
    'squeezenet1_1.onnx': 'https://gist.github.com/zhreshold/bcda4716699ac97ea44f791c24310193/raw/93672b029103648953c4e5ad3ac3aadf346a4cdc/squeezenet1_1_0.2.onnx',
    'lenet.onnx': 'https://gist.github.com/zhreshold/bcda4716699ac97ea44f791c24310193/raw/93672b029103648953c4e5ad3ac3aadf346a4cdc/lenet_0.2.onnx',
    'resnet18_1_0.onnx': 'https://gist.github.com/zhreshold/bcda4716699ac97ea44f791c24310193/raw/b385b1b242dc89a35dd808235b885ed8a19aedc1/resnet18_1.0.onnx'}

# download and add paths
for k, v  in URLS.items():
    name = k.split('.')[0]
    relpath = os.path.join('onnx', k)
    abspath = download_testdata(v, relpath, module='onnx')
    locals()[name] = abspath

# symbol for graph comparison
super_resolution_sym = get_super_resolution()
