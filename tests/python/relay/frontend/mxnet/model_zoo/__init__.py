"""MXNet and Relay model zoo."""
from __future__ import absolute_import
from . import mlp, resnet, vgg, dqn, dcgan, squeezenet, inception_v3
import tvm.relay.testing

_num_class = 1000
_batch = 2

# mlp fc
mx_mlp = mlp.get_symbol(_num_class)
relay_mlp = tvm.relay.testing.mlp.get_workload(_batch, _num_class)[0]

# vgg fc
mx_vgg = {}
relay_vgg = {}
for num_layers in [11, 13, 16, 19]:
    mx_vgg[num_layers] = vgg.get_symbol(_num_class, num_layers)
    relay_vgg[num_layers] = tvm.relay.testing.vgg.get_workload(
        _batch, _num_class, num_layers=num_layers)[0]

# resnet fc
mx_resnet = {}
relay_resnet = {}
for num_layers in [18, 34, 50, 101, 152, 200, 269]:
    mx_resnet[num_layers] = resnet.get_symbol(_num_class, num_layers, '3,224,224')
    relay_resnet[num_layers] = tvm.relay.testing.resnet.get_workload(
        _batch, _num_class, num_layers=num_layers)[0]

# squeezenet
mx_squeezenet = {}
relay_squeezenet = {}
for version in ['1.0', '1.1']:
    mx_squeezenet[version] = squeezenet.get_symbol(version=version)
    relay_squeezenet[version] = tvm.relay.testing.squeezenet.get_workload(_batch, version=version)[0]

# inception
mx_inception_v3 = inception_v3.get_symbol()
relay_inception_v3 = tvm.relay.testing.inception_v3.get_workload(_batch)[0]

# dqn
mx_dqn = dqn.get_symbol()
relay_dqn = tvm.relay.testing.dqn.get_workload(_batch)[0]

# dcgan generator
mx_dcgan = dcgan.get_symbol()
relay_dcgan = tvm.relay.testing.dcgan.get_workload(_batch)[0]
