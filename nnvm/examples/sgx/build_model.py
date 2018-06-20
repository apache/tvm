"""Creates a neural network graph module, the system library, and params.
Heavily inspired by tutorials/from_mxnet.py
"""
from __future__ import print_function
import ast
import os
from os import path as osp
import tempfile

import mxnet as mx
from mxnet.gluon.model_zoo.vision import get_model
from mxnet.gluon.utils import download
import nnvm
import nnvm.compiler
import numpy as np
from PIL import Image
import tvm


EXAMPLE_ROOT = osp.abspath(osp.join(osp.dirname(__file__)))
BIN_DIR = osp.join(EXAMPLE_ROOT, 'bin')
LIB_DIR = osp.join(EXAMPLE_ROOT, 'lib')

TVM_TARGET = 'llvm --system-lib'


def _download_model_and_image(out_dir):
    mx_model = get_model('resnet18_v1', pretrained=True)

    img_path = osp.join(out_dir, 'cat.png')
    bin_img_path = osp.join(out_dir, 'cat.bin')
    download(
        'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true',
        img_path)
    img = Image.open(img_path).resize((224, 224))
    img = _transform_image(img)
    img.astype('float32').tofile(bin_img_path)
    shape_dict = {'data': img.shape}

    return mx_model, shape_dict


def _transform_image(image):
    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image


def main():
    # load the model, input image, and imagenet classes
    mx_model, shape_dict = _download_model_and_image(BIN_DIR)

    # convert the model, add a softmax
    sym, params = nnvm.frontend.from_mxnet(mx_model)
    sym = nnvm.sym.softmax(sym)

    # build the graph
    graph, lib, params = nnvm.compiler.build(
        sym, TVM_TARGET, shape_dict, params=params)

    # save the built graph
    if not osp.isdir(LIB_DIR):
        os.mkdir(LIB_DIR)
    lib.save(osp.join(LIB_DIR, 'deploy_lib.o'))
    with open(osp.join(LIB_DIR, 'deploy_graph.json'), 'w') as f_graph_json:
        f_graph_json.write(graph.json())
    with open(osp.join(LIB_DIR, 'deploy_params.bin'), 'wb') as f_params:
        f_params.write(nnvm.compiler.save_param_dict(params))


if __name__ == '__main__':
    main()
