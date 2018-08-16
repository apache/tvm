"""MXNet and NNVM model zoo."""
from __future__ import absolute_import
from . import mlp, resnet, vgg, dqn, dcgan, squeezenet, inception_v3
import nnvm.testing

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

# squeezenet
mx_squeezenet = {}
nnvm_squeezenet = {}
for version in ['1.0', '1.1']:
    mx_squeezenet[version] = squeezenet.get_symbol(version=version)
    nnvm_squeezenet[version] = nnvm.testing.squeezenet.get_workload(1, version=version)[0]

# inception
mx_inception_v3 = inception_v3.get_symbol()
nnvm_inception_v3 = nnvm.testing.inception_v3.get_workload(1)[0]

# dqn
mx_dqn = dqn.get_symbol()
nnvm_dqn = nnvm.testing.dqn.get_workload(1)[0]

# dcgan generator
mx_dcgan = dcgan.get_symbol()
nnvm_dcgan = nnvm.testing.dcgan.get_workload(1)[0]
