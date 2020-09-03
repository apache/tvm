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
# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name, W0611, C0413
"""Vitis-AI codegen tests."""
import sys
import numpy as np

import pytest
pytest.importorskip('pyxir')
import pyxir.contrib.target.DPUCADX8G

import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.op.contrib.vitis_ai import annotation
from tvm.relay.build_module import bind_params_by_name
from tvm.contrib.target import vitis_ai

def set_func_attr(func, compile_name, symbol_name):
    func = func.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
    func = func.with_attr("Inline", tvm.tir.IntImm("int32", 1))
    func = func.with_attr("Compiler", compile_name)
    func = func.with_attr("global_symbol", symbol_name)
    return func

def _create_graph():
    shape = (10, 10)
    mod = tvm.IRModule()
    x = relay.var('x', shape=shape)
    y = relay.var('y', shape=shape)
    z = x + x
    p = y * y
    func = relay.Function([x, y], p - z)
    mod["main"] = func
    params = {}
    params["x"] = np.random.rand(10, 10).astype('float32')
    params["y"] = np.random.rand(10, 10).astype('float32')
    return mod, params


def _construct_model(func, params=None):
    mod = tvm.IRModule()
    mod["main"] = func
    if params is None:
        params = {}
    mod["main"] = bind_params_by_name(mod["main"], params)
    mod = annotation(mod, params, "DPUCADX8G")
    mod = transform.MergeCompilerRegions()(mod)
    mod = transform.PartitionGraph()(mod)
    fcompile = tvm._ffi.get_global_func("relay.ext.vitis_ai")
    subgraph_mod = tvm.IRModule()
    for _, funcnode in mod.functions.items():
        if funcnode.attrs and 'Compiler' in funcnode.attrs and \
           funcnode.attrs['Compiler'] == 'vitis_ai':
            subgraph_mod["main"] = funcnode
            with tvm.transform.PassContext(opt_level=3, \
                                           config={'relay.ext.vitis_ai.options.target':
                                                   'DPUCADX8G'}):
                fcompile(subgraph_mod["main"])


def test_add():
    shape = (10, 10)
    x = relay.var('x', shape=shape)
    y = x + x
    func = relay.Function([x], y)
    _construct_model(func)

def test_relu():
    shape = (10, 10)
    x = relay.var('x', shape=shape)
    y = relay.nn.relu(x)
    func = relay.Function([x], y)
    _construct_model(func)

def test_conv2d():
    x = relay.var('x', shape=(1, 3, 224, 224))
    w = relay.const(np.zeros((16, 3, 3, 3), dtype='float32'))
    y = relay.nn.conv2d(x, w, strides=[2, 2], padding=[1, 1, 1, 1], kernel_size=[3, 3])
    func = relay.Function([x], y)
    params = {}
    params["x"] = np.zeros((1, 3, 224, 224), dtype='float32')
    params["w"] = np.random.rand(16, 3, 3, 3).astype('float32')
    _construct_model(func, params)

def test_batchnorm():
    data = relay.var('data', shape=(1, 16, 112, 112))
    bn_gamma = relay.var("bn_gamma", relay.TensorType((16, ), "float32"))
    bn_beta = relay.var("bn_beta", relay.TensorType((16, ), "float32"))
    bn_mmean = relay.var("bn_mean", relay.TensorType((16, ), "float32"))
    bn_mvar = relay.var("bn_var", relay.TensorType((16, ), "float32"))
    bn_output = relay.nn.batch_norm(data, bn_gamma, bn_beta, bn_mmean,
                                    bn_mvar)
    func = relay.Function([data, bn_gamma, bn_beta, bn_mmean,
                           bn_mvar], bn_output[0])
    params = {}
    params["data"] = np.zeros((1, 16, 112, 112), dtype='float32')
    params["bn_gamma"] = np.random.rand(16).astype('float32')
    params["bn_beta"] = np.random.rand(16).astype('float32')
    params["bn_mean"] = np.random.rand(16).astype('float32')
    params["bn_var"] = np.random.rand(16).astype('float32')
    _construct_model(func, params)

def test_global_avg_pool2d():
    shape = (10, 10, 10, 10)
    x = relay.var('x', shape=shape)
    y = relay.nn.global_avg_pool2d(x)
    func = relay.Function([x], y)
    _construct_model(func)

def test_avg_pool2d():
    shape = (10, 10, 10, 10)
    x = relay.var('x', shape=shape)
    y = relay.nn.avg_pool2d(x, pool_size=(3, 3))
    func = relay.Function([x], y)
    _construct_model(func)

def test_annotate():
    """Test annotation with Vitis-AI DP (DPUCADX8G)"""
    def partition():
        data = relay.var("data", relay.TensorType((1, 3, 224, 224), "float32"))
        weight = relay.var("weight", relay.TensorType((16, 3, 3, 3), "float32"))
        bn_gamma = relay.var("bn_gamma", relay.TensorType((16, ), "float32"))
        bn_beta = relay.var("bn_beta", relay.TensorType((16, ), "float32"))
        bn_mmean = relay.var("bn_mean", relay.TensorType((16, ), "float32"))
        bn_mvar = relay.var("bn_var", relay.TensorType((16, ), "float32"))

        conv = relay.nn.conv2d(
            data=data,
            weight=weight,
            kernel_size=(3, 3),
            channels=16,
            padding=(1, 1))
        bn_output = relay.nn.batch_norm(conv, bn_gamma, bn_beta, bn_mmean,
                                        bn_mvar)

        func = relay.Function([data, weight, bn_gamma, bn_beta, bn_mmean,
                               bn_mvar], bn_output.astuple())
        mod = tvm.IRModule()
        mod["main"] = func
        params = {}
        params["weight"] = np.random.rand(16, 3, 3, 3).astype('float32')
        params["bn_gamma"] = np.random.rand(16).astype('float32')
        params["bn_beta"] = np.random.rand(16).astype('float32')
        params["bn_mean"] = np.random.rand(16).astype('float32')
        params["bn_var"] = np.random.rand(16).astype('float32')
        mod = annotation(mod, params, "DPUCADX8G")

        opt_pass = tvm.transform.Sequential([
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
        ])

        with tvm.transform.PassContext(opt_level=3):
            mod = opt_pass(mod)

        return mod

    def expected():
        # function variables for conv2d
        data0 = relay.var("data0", relay.TensorType((1, 3, 224, 224), "float32"))
        weight0 = relay.var("weight0", relay.TensorType((16, 3, 3, 3), "float32"))
        conv = relay.nn.conv2d(
            data=data0,
            weight=weight0,
            kernel_size=(3, 3),
            channels=16,
            padding=(1, 1))

        # function variables for batch_norm
        bn_gamma0 = relay.var("bn_gamma0", relay.TensorType((16, ), "float32"))
        bn_beta0 = relay.var("bn_beta0", relay.TensorType((16, ), "float32"))
        bn_mmean0 = relay.var("bn_mean0", relay.TensorType((16, ), "float32"))
        bn_mvar0 = relay.var("bn_var0", relay.TensorType((16, ), "float32"))
        bn = relay.nn.batch_norm(conv, bn_gamma0, bn_beta0, bn_mmean0, bn_mvar0)
        func0 = relay.Function([data0, weight0, bn_gamma0, bn_beta0, bn_mmean0, bn_mvar0],
                               bn.astuple())
        func0 = set_func_attr(func0, "vitis_ai", "vitis_ai_0")
        gv0 = relay.GlobalVar("vitis_ai_0")
        mod = tvm.IRModule()
        mod[gv0] = func0

        # main function
        data = relay.var("data", relay.TensorType((1, 3, 224, 224), "float32"))
        weight = relay.var("weight", relay.TensorType((16, 3, 3, 3), "float32"))
        bn_gamma = relay.var("bn_gamma", relay.TensorType((16, ), "float32"))
        bn_beta = relay.var("bn_beta", relay.TensorType((16, ), "float32"))
        bn_mmean = relay.var("bn_mean", relay.TensorType((16, ), "float32"))
        bn_mvar = relay.var("bn_var", relay.TensorType((16, ), "float32"))
        call0 = gv0(data, weight, bn_gamma, bn_beta, bn_mmean, bn_mvar)
        mod["main"] = relay.Function([data, weight, bn_gamma, bn_beta, bn_mmean,
                                      bn_mvar], call0)
        return mod

    partitioned = partition()
    ref_mod = expected()

    assert tvm.ir.structural_equal(partitioned, ref_mod, map_free_vars=True)


if __name__ == "__main__":
    if sys.platform == "win32":
        print("Skip test on Windows for now")
    else:
        test_annotate()
        test_add()
        test_relu()
        test_conv2d()
        test_batchnorm()
        test_global_avg_pool2d()
        test_avg_pool2d()
