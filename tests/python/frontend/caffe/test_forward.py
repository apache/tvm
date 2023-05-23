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
# pylint: disable=import-self, invalid-name, unused-argument, unspecified-encoding
"""
Caffe testcases
====================
This article is a test script to test Caffe operator with Relay.
"""
import os
import logging
import numpy as np
import pytest

from google.protobuf import text_format
import caffe
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2 as pb

import tvm
import tvm.testing
from tvm import relay
from tvm.contrib import graph_executor
from tvm.contrib.download import download_testdata

os.environ["GLOG_minloglevel"] = "2"

logging.basicConfig(level=logging.ERROR)

CURRENT_DIR = os.path.join(os.path.expanduser("~"), ".tvm_test_data", "caffe_test")

#######################################################################
# Generic functions for TVM & Caffe
# ------------------------------------------


def _create_dir(d_path):
    """If the directory is not existed, create it"""
    if not (os.path.exists(d_path) and os.path.isdir(d_path)):
        os.makedirs(d_path)


def _list_to_str(ll):
    """Convert list or tuple to str, separated by underline."""
    if isinstance(ll, (tuple, list)):
        tmp = [str(i) for i in ll]
        res = "_".join(tmp)
    return res


def _gen_filename_str(op_name, data_shape, *args, **kwargs):
    """Combining the filename according to the op_name, shape and other args."""
    file_dir = os.path.join(CURRENT_DIR, op_name)
    _create_dir(file_dir)
    res = op_name + "_"
    shape_str = _list_to_str(list(data_shape))
    res += shape_str
    for arg in args:
        if isinstance(arg, (tuple, list)):
            res += "_" + _list_to_str(arg)
        elif isinstance(arg, (int, float, str)):
            res += "_" + str(arg)
    for _, v in kwargs.items():
        if isinstance(v, (tuple, list)):
            res += "_" + _list_to_str(v)
        elif isinstance(v, (int, float, str)):
            res += "_" + str(v)
    res = res.replace(".", "_")
    res = res.replace("-", "_")
    proto_file = os.path.join(file_dir, res + ".prototxt")
    blob_file = os.path.join(file_dir, res + ".caffemodel")
    solver_file = os.path.join(file_dir, res + "_solver.prototxt")

    return (proto_file, blob_file, solver_file)


def _save_prototxt(n_netspec, f_path):
    """Generate .prototxt file according to caffe.NetSpec"""
    s = n_netspec.to_proto()
    with open(f_path, "w") as f:
        f.write(str(s))


def _save_solver(solver_file, proto_file, blob_file):
    """Define a solver proto, you can change the configs."""
    blob_file_prefix = blob_file.split(".caffemodel")[0]
    s = pb.SolverParameter()
    s.train_net = proto_file
    s.base_lr = 0.01
    s.momentum = 0.9
    s.weight_decay = 0.0005
    s.lr_policy = "inv"
    s.gamma = 0.0001
    s.power = 0.75
    s.display = 1
    s.max_iter = 100000
    s.snapshot = 100000
    s.snapshot_prefix = blob_file_prefix

    with open(solver_file, "w") as f:
        f.write(str(s))


def _save_caffemodel(solver_file, blob_file):
    """Generate .caffemodel file."""
    solver = caffe.SGDSolver(solver_file)
    solver.net.save(blob_file)


def _gen_model_files(n_netspec, proto_file, blob_file, solver_file):
    _save_prototxt(n_netspec, proto_file)
    _save_solver(solver_file, proto_file, blob_file)
    _save_caffemodel(solver_file, blob_file)


def _siso_op(data, func, *args, **kwargs):
    """Create single input and single output Caffe op"""
    n = caffe.NetSpec()
    n.data = L.Input(input_param={"shape": {"dim": list(data.shape)}})
    n.output = func(n.data, *args, **kwargs)
    return n


def _miso_op(data_list, func, *args, **kwargs):
    """Create multi input and single output Caffe op"""
    n = caffe.NetSpec()
    if not isinstance(data_list, (tuple, list)):
        raise TypeError(f"Need tuple or list but get {type(data_list)}")
    input_list = []
    for idx, data in enumerate(data_list):
        n["data" + str(idx)] = L.Input(input_param={"shape": {"dim": list(data.shape)}})
        input_list.append(n["data" + str(idx)])
    n.output = func(*input_list, *args, **kwargs)
    return n


def _simo_op(data, func, *args, **kwargs):
    """Create single input and multi output Caffe op"""
    n = caffe.NetSpec()
    n.data = L.Input(input_param={"shape": {"dim": list(data.shape)}})
    output_list = func(n.data, *args, **kwargs)
    for idx, out in enumerate(output_list):
        n["output" + str(idx)] = out
    return n


def _run_caffe(data, proto_file, blob_file):
    """Run caffe model by Caffe according to .caffemodel and .prototxt"""
    net = caffe.Net(proto_file, blob_file, caffe.TEST)
    if isinstance(data, (list, tuple)):
        for idx, d in enumerate(data):
            net.blobs["data" + str(idx)].data[...] = d
    else:
        net.blobs["data"].data[...] = data
    out = net.forward()

    caffe_output = []
    for i in range(len(out.keys())):
        if "output" + str(i) not in out.keys():
            caffe_output.clear()
            return list(out.values())
        caffe_output.append(out["output" + str(i)])
    return caffe_output


def _run_tvm(data, proto_file, blob_file):
    """Run caffe model by TVM according to .caffemodel and .prototxt"""
    init_net = pb.NetParameter()
    predict_net = pb.NetParameter()

    # load model
    with open(proto_file, "r") as f:
        text_format.Merge(f.read(), predict_net)
    # load blob
    with open(blob_file, "rb") as f:
        init_net.ParseFromString(f.read())

    shape_dict = {}
    dtype_dict = {}
    if isinstance(data, (tuple, list)):
        for idx, d in enumerate(data):
            shape_dict["data" + str(idx)] = d.shape
            dtype_dict["data" + str(idx)] = "float32"
    else:
        shape_dict = {"data": data.shape}
        dtype_dict = {"data": "float32"}

    mod, params = relay.frontend.from_caffe(init_net, predict_net, shape_dict, dtype_dict)

    target = "llvm"

    dev = tvm.cpu(0)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    dtype = "float32"
    m = graph_executor.GraphModule(lib["default"](dev))
    if isinstance(data, (tuple, list)):
        for idx, d in enumerate(data):
            m.set_input("data" + str(idx), tvm.nd.array(d.astype(dtype)))
    else:
        m.set_input("data", tvm.nd.array(data.astype(dtype)))
    # execute
    m.run()
    tvm_output = []
    # get outputs
    for i in range(m.get_num_outputs()):
        tvm_output.append(m.get_output(i).numpy())
    return tvm_output


def _compare_caffe_tvm(caffe_out, tvm_out, is_network=False):
    for i, _ in enumerate(caffe_out):
        if is_network:
            caffe_out[i] = caffe_out[i][:1]
        tvm.testing.assert_allclose(caffe_out[i], tvm_out[i], rtol=1e-5, atol=1e-5)


def _test_op(data, func_op, op_name, **kwargs):
    """Single op testing pipline."""
    shape_list = []
    if isinstance(data, (list, tuple)):
        n = _miso_op(data, func_op, **kwargs)
        for d in data:
            shape_list.extend(list(d.shape))
    else:
        output_num = 1
        if "ntop" in kwargs:
            output_num = kwargs["ntop"]
        if output_num == 1:
            n = _siso_op(data, func_op, **kwargs)
        else:
            n = _simo_op(data, func_op, **kwargs)
        shape_list = list(data.shape)

    # obtain the .caffemodel file and .prototxt file
    (proto_file, blob_file, solver_file) = _gen_filename_str(op_name, shape_list, **kwargs)
    _gen_model_files(n, proto_file, blob_file, solver_file)
    # run model in Caffe
    caffe_out = _run_caffe(data, proto_file, blob_file)
    # run model in TVM
    tvm_out = _run_tvm(data, proto_file, blob_file)
    _compare_caffe_tvm(caffe_out, tvm_out)


def _test_network(data, proto_file, blob_file):
    # run model in Caffe
    caffe_out = _run_caffe(data, proto_file, blob_file)
    # run model in TVM
    tvm_out = _run_tvm(data, proto_file, blob_file)
    _compare_caffe_tvm(caffe_out, tvm_out, is_network=True)


#######################################################################
# BatchNorm
# -----------


def _test_batchnorm(data, moving_average_fraction=0.999, eps=1e-5):
    """One iteration of BatchNorm"""
    _test_op(
        data, L.BatchNorm, "BatchNorm", moving_average_fraction=moving_average_fraction, eps=eps
    )


def test_forward_BatchNorm():
    """BatchNorm"""
    data = np.random.rand(1, 3, 10, 10).astype(np.float32)
    _test_batchnorm(data)
    _test_batchnorm(data, moving_average_fraction=0.88, eps=1e-4)


#######################################################################
# Concat
# -----------


def _test_concat(data_list, axis=1):
    """One iteration of Concat"""
    _test_op(data_list, L.Concat, "Concat", axis=axis)


def test_forward_Concat():
    """Concat"""
    _test_concat([np.random.rand(1, 3, 10, 10), np.random.rand(1, 2, 10, 10)], axis=1)
    _test_concat([np.random.rand(3, 10, 10), np.random.rand(2, 10, 10)], axis=0)
    _test_concat([np.random.rand(3, 10), np.random.rand(2, 10)], axis=0)


#######################################################################
# Convolution
# -----------


def _test_convolution(data, **kwargs):
    """One iteration of Convolution"""
    _test_op(data, L.Convolution, "Convolution", **kwargs)


def test_forward_Convolution():
    """Convolution"""
    data = np.random.rand(1, 3, 10, 10).astype(np.float32)
    _test_convolution(
        data,
        num_output=20,
        bias_term=True,
        pad=0,
        kernel_size=3,
        stride=2,
        dilation=1,
        weight_filler=dict(type="xavier"),
        bias_filler=dict(type="xavier"),
    )
    _test_convolution(
        data,
        num_output=20,
        bias_term=False,
        pad=[1, 2],
        kernel_size=3,
        stride=2,
        dilation=1,
        weight_filler=dict(type="xavier"),
        bias_filler=dict(type="xavier"),
    )
    _test_convolution(
        data,
        num_output=20,
        bias_term=True,
        pad=[1, 2],
        kernel_size=[3, 5],
        stride=[2, 1],
        dilation=[1, 2],
        weight_filler=dict(type="xavier"),
        bias_filler=dict(type="xavier"),
    )
    _test_convolution(
        np.random.rand(1, 2, 10, 10).astype(np.float32),
        num_output=20,
        bias_term=True,
        pad=[1, 2],
        kernel_size=[3, 5],
        stride=[2, 1],
        dilation=[1, 2],
        weight_filler=dict(type="xavier"),
        bias_filler=dict(type="xavier"),
        group=2,
    )
    _test_convolution(
        data,
        num_output=20,
        bias_term=True,
        pad_h=1,
        pad_w=2,
        kernel_h=3,
        kernel_w=5,
        stride_h=2,
        stride_w=1,
        dilation=[1, 2],
        weight_filler=dict(type="xavier"),
        bias_filler=dict(type="xavier"),
    )


#######################################################################
# Crop
# -----------


def _test_crop(data, **kwargs):
    """One iteration of Crop"""
    _test_op(data, L.Crop, "Crop", **kwargs)


def test_forward_Crop():
    """Crop"""
    _test_crop([np.random.rand(10, 10, 120, 120), np.random.rand(10, 5, 50, 60)])
    _test_crop([np.random.rand(10, 10, 120, 120), np.random.rand(10, 5, 50, 60)], axis=1)
    _test_crop([np.random.rand(10, 10, 120, 120), np.random.rand(10, 5, 50, 60)], axis=1, offset=2)
    _test_crop(
        [np.random.rand(10, 10, 120, 120), np.random.rand(10, 5, 50, 60)], axis=1, offset=[1, 2, 4]
    )
    _test_crop(
        [np.random.rand(10, 10, 120, 120), np.random.rand(10, 5, 50, 60)], axis=2, offset=[2, 4]
    )
    _test_crop([np.random.rand(10, 120, 120), np.random.rand(5, 50, 60)], axis=1, offset=[2, 4])
    _test_crop([np.random.rand(120, 120), np.random.rand(50, 60)], axis=0, offset=[2, 4])


#######################################################################
# Deconvolution
# -----------


def _test_deconvolution(data, **kwargs):
    """One iteration of Deconvolution"""
    _test_op(data, L.Deconvolution, "Deconvolution", **kwargs)


def test_forward_Deconvolution():
    """Deconvolution"""
    data = np.random.rand(1, 16, 32, 32).astype(np.float32)
    _test_deconvolution(
        data,
        convolution_param=dict(
            num_output=20,
            bias_term=True,
            pad=0,
            kernel_size=3,
            stride=2,
            dilation=1,
            weight_filler=dict(type="xavier"),
            bias_filler=dict(type="xavier"),
        ),
    )
    _test_deconvolution(
        data,
        convolution_param=dict(
            num_output=20,
            bias_term=False,
            pad=[1, 2],
            kernel_size=3,
            stride=2,
            dilation=1,
            weight_filler=dict(type="xavier"),
            bias_filler=dict(type="xavier"),
        ),
    )
    _test_deconvolution(
        data,
        convolution_param=dict(
            num_output=20,
            bias_term=True,
            pad_h=1,
            pad_w=2,
            kernel_h=3,
            kernel_w=5,
            stride_h=2,
            stride_w=1,
            dilation=1,
            weight_filler=dict(type="xavier"),
            bias_filler=dict(type="xavier"),
        ),
    )
    _test_deconvolution(
        data,
        convolution_param=dict(
            num_output=16,
            bias_term=False,
            pad=0,
            kernel_size=2,
            stride=2,
            dilation=1,
            group=16,
            weight_filler=dict(type="xavier"),
            bias_filler=dict(type="xavier"),
        ),
    )
    data = np.random.rand(1, 100, 32, 32).astype(np.float32)
    _test_deconvolution(
        data,
        convolution_param=dict(
            num_output=100,
            bias_term=False,
            pad=0,
            kernel_size=2,
            stride=2,
            dilation=1,
            group=100,
            weight_filler=dict(type="xavier"),
            bias_filler=dict(type="xavier"),
        ),
    )


#######################################################################
# Dropout
# -----------


def _test_dropout(data, **kwargs):
    """One iteration of Dropout"""
    _test_op(data, L.Dropout, "Dropout", **kwargs)


def test_forward_Dropout():
    """Dropout"""
    data = np.random.rand(1, 3, 10, 10).astype(np.float32)
    _test_dropout(data)
    _test_dropout(data, dropout_ratio=0.7)


#######################################################################
# Eltwise
# -----------


def _test_eltwise(data_list, **kwargs):
    """One iteration of Eltwise"""
    _test_op(data_list, L.Eltwise, "Eltwise", **kwargs)


def test_forward_Eltwise():
    """Eltwise"""
    _test_eltwise(
        [
            np.random.rand(1, 3, 10, 11).astype(np.float32),
            np.random.rand(1, 3, 10, 11).astype(np.float32),
        ],
        operation=0,
    )
    _test_eltwise(
        [
            np.random.rand(1, 3, 10, 11).astype(np.float32),
            np.random.rand(1, 3, 10, 11).astype(np.float32),
        ],
        operation=1,
    )
    _test_eltwise(
        [
            np.random.rand(1, 3, 10, 11).astype(np.float32),
            np.random.rand(1, 3, 10, 11).astype(np.float32),
        ],
        operation=2,
    )
    _test_eltwise(
        [
            np.random.rand(1, 3, 10, 11).astype(np.float32),
            np.random.rand(1, 3, 10, 11).astype(np.float32),
        ],
        operation=1,
        coeff=[0.5, 1],
    )
    _test_eltwise(
        [
            np.random.rand(1, 3, 10, 11).astype(np.float32),
            np.random.rand(1, 3, 10, 11).astype(np.float32),
            np.random.rand(1, 3, 10, 11).astype(np.float32),
        ],
        operation=0,
    )
    _test_eltwise(
        [
            np.random.rand(1, 3, 10, 11).astype(np.float32),
            np.random.rand(1, 3, 10, 11).astype(np.float32),
            np.random.rand(1, 3, 10, 11).astype(np.float32),
            np.random.rand(1, 3, 10, 11).astype(np.float32),
        ],
        operation=1,
    )
    _test_eltwise(
        [
            np.random.rand(1, 3, 10, 11).astype(np.float32),
            np.random.rand(1, 3, 10, 11).astype(np.float32),
            np.random.rand(1, 3, 10, 11).astype(np.float32),
            np.random.rand(1, 3, 10, 11).astype(np.float32),
            np.random.rand(1, 3, 10, 11).astype(np.float32),
        ],
        operation=2,
    )
    _test_eltwise(
        [
            np.random.rand(1, 3, 10, 11).astype(np.float32),
            np.random.rand(1, 3, 10, 11).astype(np.float32),
            np.random.rand(1, 3, 10, 11).astype(np.float32),
            np.random.rand(1, 3, 10, 11).astype(np.float32),
            np.random.rand(1, 3, 10, 11).astype(np.float32),
            np.random.rand(1, 3, 10, 11).astype(np.float32),
        ],
        operation=1,
        coeff=[0.5, 1, 0.2, 1.8, 3.1, 0.1],
    )


#######################################################################
# Flatten
# -----------


def _test_flatten(data, axis=1):
    """One iteration of Flatten"""
    _test_op(data, L.Flatten, "Flatten", axis=axis)


def test_forward_Flatten():
    """Flatten"""
    data = np.random.rand(1, 3, 10, 10).astype(np.float32)
    _test_flatten(data)
    _test_flatten(data, axis=1)


#######################################################################
# Flatten
# -----------


def _test_inner_product(data, **kwargs):
    """One iteration of InnerProduct"""
    _test_op(data, L.InnerProduct, "InnerProduct", **kwargs)


def test_forward_InnerProduct():
    """InnerProduct"""
    data = np.random.rand(1, 3, 10, 10)
    _test_inner_product(data, num_output=20, bias_term=False, weight_filler=dict(type="xavier"))
    _test_inner_product(
        data,
        num_output=20,
        bias_term=True,
        weight_filler=dict(type="xavier"),
        bias_filler=dict(type="xavier"),
    )
    _test_inner_product(
        np.random.rand(20, 10).astype(np.float32),
        num_output=30,
        bias_term=True,
        weight_filler=dict(type="xavier"),
        bias_filler=dict(type="xavier"),
    )


#######################################################################
# LRN
# -----------


def _test_lrn(data, local_size=5, alpha=1.0, beta=0.75, k=1.0):
    """One iteration of LRN"""
    _test_op(data, L.LRN, "LRN", local_size=local_size, alpha=alpha, beta=beta, k=k)


def test_forward_LRN():
    """LRN"""
    data = np.random.rand(1, 3, 10, 10).astype(np.float32)
    _test_lrn(data)
    _test_lrn(data, local_size=3)
    _test_lrn(data, local_size=3, alpha=2.0)
    _test_lrn(
        data,
        local_size=3,
        alpha=2.0,
        beta=0.5,
    )
    _test_lrn(data, local_size=3, alpha=2.0, beta=0.5, k=2.0)


#######################################################################
# Permute
# -------


def _test_permute(data, **kwargs):
    """One iteration of Permute."""
    _test_op(data, L.Permute, "Permute", **kwargs)


def test_forward_Permute():
    """Permute"""
    data = np.random.rand(2, 3, 4).astype(np.float32)
    _test_permute(data, permute_param={"order": [0, 1, 2]})
    _test_permute(data, permute_param={"order": [0, 2, 1]})
    _test_permute(data, permute_param={"order": [1, 0, 2]})
    _test_permute(data, permute_param={"order": [1, 2, 0]})
    _test_permute(data, permute_param={"order": [2, 0, 1]})
    _test_permute(data, permute_param={"order": [2, 1, 0]})


#######################################################################
# Pooling
# -----------


def _test_pooling(data, **kwargs):
    """One iteration of Pooling."""
    _test_op(data, L.Pooling, "Pooling", **kwargs)


def test_forward_Pooling():
    """Pooing"""
    data = np.random.rand(1, 3, 10, 10).astype(np.float32)
    # MAX Pooling
    _test_pooling(data, kernel_size=2, stride=2, pad=0, pool=P.Pooling.MAX)
    _test_pooling(
        data, kernel_h=2, kernel_w=3, stride_h=2, stride_w=1, pad_h=1, pad_w=2, pool=P.Pooling.MAX
    )
    _test_pooling(data, pool=P.Pooling.MAX, global_pooling=True)

    # AVE Pooing
    _test_pooling(data, kernel_size=2, stride=2, pad=0, pool=P.Pooling.AVE)
    _test_pooling(
        data, kernel_h=2, kernel_w=3, stride_h=2, stride_w=1, pad_h=1, pad_w=2, pool=P.Pooling.AVE
    )
    _test_pooling(data, pool=P.Pooling.AVE, global_pooling=True)


#######################################################################
# Power
# -----
def _test_power(data, **kwargs):
    """One iteration of Power."""
    _test_op(data, L.Power, "Power", **kwargs)


def test_forward_Power():
    """Power"""
    data = np.random.rand(1, 3, 10, 10).astype(np.float32)
    _test_power(data, power_param={"power": 0.37, "scale": 0.83, "shift": -2.4})
    _test_power(data, power_param={"power": 0.37, "scale": 0.83, "shift": 0.0})
    _test_power(data, power_param={"power": 0.0, "scale": 0.83, "shift": -2.4})
    _test_power(data, power_param={"power": 1.0, "scale": 0.83, "shift": -2.4})
    _test_power(data, power_param={"power": 2.0, "scale": 0.34, "shift": -2.4})
    _test_power(data, power_param={"power": 1.0, "scale": 1.0, "shift": 0.0})


#######################################################################
# PReLU
# -----------


def _test_prelu(data, **kwargs):
    """One iteration of PReLU."""
    _test_op(data, L.PReLU, "PReLU", **kwargs)


def test_forward_PReLU():
    """PReLU"""
    data = np.random.rand(1, 3, 10, 10).astype(np.float32)
    _test_prelu(data, filler=dict(type="constant", value=0.5))
    _test_prelu(data)
    _test_prelu(np.random.rand(10, 20).astype(np.float32))


#######################################################################
# ReLU
# -----------


def _test_relu(data, **kwargs):
    """One iteration of ReLU."""
    _test_op(data, L.ReLU, "ReLU", **kwargs)


def test_forward_ReLU():
    """ReLU"""
    data = np.random.rand(1, 3, 10, 10).astype(np.float32)
    _test_relu(data)
    _test_relu(np.random.rand(10, 20).astype(np.float32))


#######################################################################
# Reshape
# -----------


def _test_reshape(data, **kwargs):
    """One iteration of Reshape."""
    _test_op(data, L.Reshape, "Reshape", **kwargs)


def test_forward_Reshape():
    """Reshape"""
    data = np.random.rand(1, 8, 6).astype(np.float32)
    _test_reshape(data, reshape_param={"shape": {"dim": [4, 3, 4]}})
    _test_reshape(data, reshape_param={"shape": {"dim": [2, 0, 3]}})
    _test_reshape(data, reshape_param={"shape": {"dim": [2, 0, -1]}})
    _test_reshape(data, reshape_param={"shape": {"dim": [0, -1]}})

    _test_reshape(data, reshape_param={"shape": {"dim": [2, 3]}, "axis": 2})
    _test_reshape(data, reshape_param={"shape": {"dim": [4, 3, 4]}, "axis": 1})
    _test_reshape(data, reshape_param={"shape": {"dim": [4, 3, 4]}, "axis": -3})

    _test_reshape(data, reshape_param={"shape": {"dim": [2, 4]}, "axis": 1, "num_axes": 1})
    _test_reshape(data, reshape_param={"shape": {"dim": [3, 16]}, "axis": 1, "num_axes": 2})


#######################################################################
# Scale
# -----------


def _test_scale(data, **kwargs):
    """One iteration of Scale."""
    _test_op(data, L.Scale, "Scale", **kwargs)


def test_forward_Scale():
    """Scale"""
    data = np.random.rand(1, 3, 10, 10).astype(np.float32)
    _test_scale(data, filler=dict(type="xavier"))
    _test_scale(data, filler=dict(type="xavier"), bias_term=True, bias_filler=dict(type="xavier"))


#######################################################################
# Sigmoid
# -----------


def _test_sigmoid(data, **kwargs):
    """One iteration of Sigmoid."""
    _test_op(data, L.Sigmoid, "Sigmoid", **kwargs)


def test_forward_Sigmoid():
    """Sigmoid"""
    data = np.random.rand(1, 3, 10, 10).astype(np.float32)
    _test_sigmoid(data)


#######################################################################
# Slice
# -----------


def _test_slice(data, **kwargs):
    """One iteration of Slice"""
    _test_op(data, L.Slice, "Slice", **kwargs)


def test_forward_Slice():
    """Slice"""
    data = np.random.rand(1, 3, 10, 10).astype(np.float32)
    _test_slice(data, ntop=2, slice_param=dict(axis=1, slice_point=[1]))
    _test_slice(data, ntop=2, slice_param=dict(axis=-1, slice_point=[1]))
    _test_slice(data, ntop=3, slice_param=dict(axis=2, slice_point=[1, 6]))
    _test_slice(data, ntop=3)


#######################################################################
# Softmax
# -----------


def _test_softmax(data, **kwargs):
    """One iteration of Softmax"""
    _test_op(data, L.Softmax, "Softmax", **kwargs)


def test_forward_Softmax():
    """Softmax"""
    _test_softmax(np.random.rand(1, 3, 10, 10).astype(np.float32))
    _test_softmax(np.random.rand(1, 3, 10, 10).astype(np.float32), axis=2)
    _test_softmax(np.random.rand(10, 10).astype(np.float32), axis=0)
    _test_softmax(np.random.rand(2, 10, 10).astype(np.float32), axis=1)


#######################################################################
# TanH
# -----------


def _test_tanh(data, **kwargs):
    """One iteration of TanH"""
    _test_op(data, L.TanH, "TanH", **kwargs)


def test_forward_TanH():
    """TanH"""
    _test_tanh(np.random.rand(1, 3, 10, 10).astype(np.float32))
    _test_tanh(np.random.rand(3, 10, 10).astype(np.float32))
    _test_tanh(np.random.rand(10, 10).astype(np.float32))
    _test_tanh(np.random.rand(10).astype(np.float32))


#######################################################################
# Reduction
# -----------


def _test_reduction(data, **kwargs):
    """One iteration of Reduction"""
    _test_op(data, L.Reduction, "Reduction", **kwargs)


def test_forward_Reduction():
    """Reduction"""
    reduction_op = {"SUM": 1, "ASUM": 2, "SUMSQ": 3, "MEAN": 4}
    _test_reduction(np.random.rand(10).astype(np.float32), operation=reduction_op["SUM"], axis=0)
    _test_reduction(
        np.random.rand(10, 20, 30, 40).astype(np.float32), operation=reduction_op["SUM"], axis=3
    )
    _test_reduction(
        np.random.rand(10, 20, 30, 40).astype(np.float32), operation=reduction_op["SUM"], axis=1
    )
    _test_reduction(
        np.random.rand(10).astype(np.float32), operation=reduction_op["SUM"], axis=0, coeff=0.5
    )
    _test_reduction(
        np.random.rand(10, 20, 30, 40).astype(np.float32),
        operation=reduction_op["SUM"],
        axis=3,
        coeff=5.0,
    )
    _test_reduction(np.random.rand(10).astype(np.float32), operation=reduction_op["ASUM"])
    _test_reduction(
        np.random.rand(10, 20).astype(np.float32), operation=reduction_op["ASUM"], axis=1
    )
    _test_reduction(
        np.random.rand(10, 20, 30, 40).astype(np.float32), operation=reduction_op["ASUM"], axis=3
    )
    _test_reduction(
        np.random.rand(10).astype(np.float32), operation=reduction_op["ASUM"], axis=0, coeff=0.0
    )
    _test_reduction(
        np.random.rand(10, 20, 30).astype(np.float32),
        operation=reduction_op["ASUM"],
        axis=2,
        coeff=7.0,
    )
    _test_reduction(
        np.random.rand(10, 20, 30, 40, 10).astype(np.float32),
        operation=reduction_op["ASUM"],
        axis=3,
        coeff=1.0,
    )
    _test_reduction(np.random.rand(10).astype(np.float32), operation=reduction_op["SUMSQ"], axis=0)
    _test_reduction(
        np.random.rand(10, 20, 30, 40).astype(np.float32), operation=reduction_op["SUMSQ"], axis=3
    )
    _test_reduction(
        np.random.rand(10).astype(np.float32), operation=reduction_op["SUMSQ"], axis=0, coeff=0.0
    )
    _test_reduction(
        np.random.rand(10, 20, 30, 40, 50).astype(np.float32),
        operation=reduction_op["SUMSQ"],
        axis=4,
        coeff=2.0,
    )
    _test_reduction(np.random.rand(10).astype(np.float32), operation=reduction_op["MEAN"], axis=0)
    _test_reduction(
        np.random.rand(10, 20, 30, 40).astype(np.float32), operation=reduction_op["MEAN"], axis=3
    )
    _test_reduction(
        np.random.rand(10).astype(np.float32), operation=reduction_op["MEAN"], axis=0, coeff=0.0
    )
    _test_reduction(
        np.random.rand(10, 20, 30, 40).astype(np.float32),
        operation=reduction_op["MEAN"],
        axis=3,
        coeff=2.0,
    )


#######################################################################
# Embed
# -----------


def _test_embed(data, **kwargs):
    """One iteration of Embed"""
    _test_op(data, L.Embed, "Embed", **kwargs)


def test_forward_Embed():
    """Embed"""
    k = 20
    data = list(i for i in range(k))
    np.random.shuffle(data)
    # dimension is 1
    data = np.asarray(data)
    _test_embed(
        data,
        num_output=30,
        input_dim=k,
        bias_term=True,
        weight_filler=dict(type="xavier"),
        bias_filler=dict(type="xavier"),
    )
    _test_embed(
        data,
        num_output=30,
        input_dim=k,
        bias_term=False,
        weight_filler=dict(type="xavier"),
        bias_filler=dict(type="xavier"),
    )
    # dimension is 2
    data = np.reshape(data, [4, 5])
    _test_embed(
        data,
        num_output=30,
        input_dim=k,
        bias_term=True,
        weight_filler=dict(type="xavier"),
        bias_filler=dict(type="xavier"),
    )
    _test_embed(
        data,
        num_output=30,
        input_dim=k,
        bias_term=False,
        weight_filler=dict(type="xavier"),
        bias_filler=dict(type="xavier"),
    )
    # dimension is 3
    data = np.reshape(data, [2, 2, 5])
    _test_embed(
        data,
        num_output=30,
        input_dim=k,
        bias_term=True,
        weight_filler=dict(type="xavier"),
        bias_filler=dict(type="xavier"),
    )
    _test_embed(
        data,
        num_output=30,
        input_dim=k,
        bias_term=False,
        weight_filler=dict(type="xavier"),
        bias_filler=dict(type="xavier"),
    )
    # dimension is 4
    data = np.reshape(data, [2, 2, 5, 1])
    _test_embed(
        data,
        num_output=30,
        input_dim=k,
        bias_term=True,
        weight_filler=dict(type="xavier"),
        bias_filler=dict(type="xavier"),
    )
    _test_embed(
        data,
        num_output=30,
        input_dim=k,
        bias_term=False,
        weight_filler=dict(type="xavier"),
        bias_filler=dict(type="xavier"),
    )


#######################################################################
# Mobilenetv2
# -----------


def _test_mobilenetv2(data):
    """One iteration of Mobilenetv2"""
    mean_val = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    mean_val = np.reshape(mean_val, (1, 3, 1, 1))
    mean_val = np.tile(mean_val, (1, 1, 224, 224))
    data_process = data - mean_val
    data_process = data_process / 58.8
    data_process = data_process.astype(np.float32)

    proto_file_url = (
        "https://github.com/shicai/MobileNet-Caffe/raw/master/mobilenet_v2_deploy.prototxt"
    )
    blob_file_url = (
        "https://github.com/shicai/MobileNet-Caffe/blob/master/mobilenet_v2.caffemodel?raw=true"
    )
    proto_file = download_testdata(proto_file_url, "mobilenetv2.prototxt", module="model")
    blob_file = download_testdata(blob_file_url, "mobilenetv2.caffemodel", module="model")
    _test_network(data_process, proto_file, blob_file)


def test_forward_Mobilenetv2():
    """Mobilenetv2"""
    data = np.random.randint(0, 256, size=(1, 3, 224, 224)).astype(np.float32)
    _test_mobilenetv2(data)


#######################################################################
# Alexnet
# -----------


def _test_alexnet(data):
    """One iteration of Alexnet"""
    mean_val = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    mean_val = np.reshape(mean_val, (1, 3, 1, 1))
    mean_val = np.tile(mean_val, (1, 1, 227, 227))
    data_process = data - mean_val
    data_process = data_process.astype(np.float32)

    proto_file_url = (
        "https://github.com/BVLC/caffe/raw/master/models/" + "bvlc_alexnet/deploy.prototxt"
    )
    blob_file_url = "http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel"
    proto_file = download_testdata(proto_file_url, "alexnet.prototxt", module="model")
    blob_file = download_testdata(blob_file_url, "alexnet.caffemodel", module="model")
    _test_network(data_process, proto_file, blob_file)


@pytest.mark.skip(reason="See https://github.com/apache/tvm/issues/13227")
def test_forward_Alexnet():
    """Alexnet"""
    data = np.random.randint(0, 256, size=(1, 3, 227, 227)).astype(np.float32)
    _test_alexnet(data)


#######################################################################
# Resnet50
# -----------


def _test_resnet50(data):
    """One iteration of Resnet50"""
    mean_val = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    mean_val = np.reshape(mean_val, (1, 3, 1, 1))
    mean_val = np.tile(mean_val, (1, 1, 224, 224))
    data_process = data - mean_val
    data_process = data_process.astype(np.float32)

    proto_file_url = (
        "https://github.com/fernchen/CaffeModels/raw/master/resnet/ResNet-50-deploy.prototxt"
    )
    blob_file_url = (
        "https://github.com/fernchen/CaffeModels/raw/master/resnet/ResNet-50-model.caffemodel"
    )

    proto_file = download_testdata(proto_file_url, "resnet50.prototxt", module="model")
    blob_file = download_testdata(blob_file_url, "resnet50.caffemodel", module="model")

    _test_network(data_process, proto_file, blob_file)


def test_forward_Resnet50():
    """Resnet50"""
    data = np.random.randint(0, 256, size=(1, 3, 224, 224)).astype(np.float32)
    _test_resnet50(data)


#######################################################################
# Inceptionv4
# -----------


def _test_inceptionv1(data):
    """One iteration of Inceptionv4"""
    mean_val = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    mean_val = np.reshape(mean_val, (1, 3, 1, 1))
    mean_val = np.tile(mean_val, (1, 1, 224, 224))
    data_process = data - mean_val
    data_process = data_process / 58.8
    data_process = data_process.astype(np.float32)

    proto_file_url = (
        "https://github.com/BVLC/caffe/raw/master/models" + "/bvlc_googlenet/deploy.prototxt"
    )
    blob_file_url = "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel"
    proto_file = download_testdata(proto_file_url, "inceptionv1.prototxt", module="model")
    blob_file = download_testdata(blob_file_url, "inceptionv1.caffemodel", module="model")
    _test_network(data_process, proto_file, blob_file)


@pytest.mark.skip(reason="See issue https://github.com/apache/tvm/issues/13227")
def test_forward_Inceptionv1():
    """Inceptionv4"""
    data = np.random.randint(0, 256, size=(1, 3, 224, 224)).astype(np.float32)
    _test_inceptionv1(data)


if __name__ == "__main__":
    tvm.testing.main()
