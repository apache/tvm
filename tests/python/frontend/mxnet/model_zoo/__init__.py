# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""MXNet model zoo for testing purposes."""
from __future__ import absolute_import
from . import mlp, vgg, resnet, dqn, inception_v3, squeezenet, dcgan
import tvm.relay.testing

# mlp
def mx_mlp():
    num_class = 10
    return mlp.get_symbol(num_class)


def relay_mlp():
    num_class = 10
    return tvm.relay.testing.mlp.get_workload(1, num_class)[0]


# vgg
def mx_vgg(num_layers):
    num_class = 1000
    return vgg.get_symbol(num_class, num_layers)


def relay_vgg(num_layers):
    num_class = 1000
    return tvm.relay.testing.vgg.get_workload(1, num_class, num_layers=num_layers)[0]


# resnet
def mx_resnet(num_layers):
    num_class = 1000
    return resnet.get_symbol(num_class, num_layers, "3,224,224")


def relay_resnet(num_layers):
    num_class = 1000
    return tvm.relay.testing.resnet.get_workload(1, num_class, num_layers=num_layers)[0]


# dqn
mx_dqn = dqn.get_symbol


def relay_dqn():
    return tvm.relay.testing.dqn.get_workload(1)[0]


# squeezenet
def mx_squeezenet(version):
    return squeezenet.get_symbol(version=version)


def relay_squeezenet(version):
    return tvm.relay.testing.squeezenet.get_workload(1, version=version)[0]


# inception
mx_inception_v3 = inception_v3.get_symbol


def relay_inception_v3():
    return tvm.relay.testing.inception_v3.get_workload(1)[0]


# dcgan generator
mx_dcgan = dcgan.get_symbol


def relay_dcgan(batch_size):
    return tvm.relay.testing.dcgan.get_workload(batch_size=batch_size)[0]
