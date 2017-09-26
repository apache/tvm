"""MXNet and NNVM model zoo."""
from __future__ import absolute_import
from . import mlp, resnet, vgg
import nnvm.testing

__all__ = ['mx_mlp', 'nnvm_mlp', 'mx_resnet', 'nnvm_resnet', 'mx_vgg', 'nnvm_vgg']

_num_class = 1000

# mlp fc
mx_mlp = mlp.get_symbol(_num_class)
nnvm_mlp = nnvm.testing.mlp.get_workload(1, _num_class)[0]

# resnet fc
mx_resnet = {}
nnvm_resnet = {}
for num_layer in [18, 34, 50, 101, 152, 200, 269]:
    mx_resnet[num_layer] = resnet.get_symbol(_num_class, num_layer, '3,224,224')
    nnvm_resnet[num_layer] = nnvm.testing.resnet.get_workload(
        1, _num_class, num_layers=num_layer)[0]

# vgg fc
mx_vgg = {}
nnvm_vgg = {}
for num_layer in [11, 13, 16, 19]:
    mx_vgg[num_layer] = vgg.get_symbol(_num_class, num_layer)
    nnvm_vgg[num_layer] = nnvm.testing.vgg.get_workload(
        1, _num_class, num_layers=num_layer)[0]
