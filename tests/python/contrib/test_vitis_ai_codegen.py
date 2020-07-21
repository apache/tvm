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
# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name, W0611
"""Vitis-AI codegen tests."""

import numpy as np

import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.op.contrib.vitis_ai import annotation
from tvm.contrib.target import vitis_ai

import pyxir
import pyxir.contrib.target.DPUCADX8G

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
    mod = annotation(mod, params, "DPUCADX8G")
    mod = transform.MergeCompilerRegions()(mod)
    mod = transform.PartitionGraph()(mod)
    fcompile = tvm._ffi.get_global_func("relay.ext.vai")
    subgraph_mod = tvm.IRModule()
    for _, funcnode in mod.functions.items():
        if funcnode.attrs and 'Compiler' in funcnode.attrs and \
           funcnode.attrs['Compiler'] == 'vai':
            subgraph_mod["main"] = funcnode
            with tvm.transform.PassContext(opt_level=3, config={'target_':'DPUCADX8G'}):
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
    params["x"] = np.zeros((16, 3, 3, 3), dtype='float32')
    _construct_model(func, params)


def test_global_avg_pool2d():
    shape = (10, 10, 10, 10)
    x = relay.var('x', shape=shape)
    y = relay.nn.global_avg_pool2d(x)
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
            transform.InferType(),
            transform.PartitionGraph(),
            transform.SimplifyInference(),
            transform.FoldConstant(),
            transform.AlterOpLayout(),
        ])

        with tvm.transform.PassContext(opt_level=3):
            mod = opt_pass(mod)

        return mod

    def expected():
        # function for batch_norm
        data0 = relay.var("data0", relay.TensorType((1, 16, 224, 224),
                                                    "float32"))
        mod = tvm.IRModule()
        bn_gamma = relay.var("bn_gamma1", relay.TensorType((16, ), "float32"))
        bn_beta = relay.var("bn_beta1", relay.TensorType((16, ), "float32"))
        bn_mmean = relay.var("bn_mean1", relay.TensorType((16, ), "float32"))
        bn_mvar = relay.var("bn_var1", relay.TensorType((16, ), "float32"))

        bn = relay.nn.batch_norm(data0, bn_gamma, bn_beta, bn_mmean, bn_mvar)
        func0 = relay.Function([data0, bn_gamma, bn_beta, bn_mmean, bn_mvar],
                               bn.astuple())
        func0 = set_func_attr(func0, "vai", "vai_2")
        gv0 = relay.GlobalVar("vai_2")
        mod[gv0] = func0

        # function for conv2d
        data1 = relay.var("data1", relay.TensorType((1, 3, 224, 224), "float32"))
        weight1 = relay.var("weight1", relay.TensorType((16, 3, 3, 3), "float32"))
        conv = relay.nn.conv2d(
            data=data1,
            weight=weight1,
            kernel_size=(3, 3),
            channels=16,
            padding=(1, 1))
        func1 = relay.Function([data1, weight1], conv)
        func1 = set_func_attr(func1, "vai", "vai_0")
        gv1 = relay.GlobalVar("vai_0")
        mod[gv1] = func1

        # main function
        data = relay.var("data", relay.TensorType((1, 3, 224, 224), "float32"))
        weight = relay.var("weight", relay.TensorType((16, 3, 3, 3), "float32"))
        bn_gamma0 = relay.var("bn_gamma", relay.TensorType((16, ), "float32"))
        bn_beta0 = relay.var("bn_beta", relay.TensorType((16, ), "float32"))
        bn_mmean0 = relay.var("bn_mean", relay.TensorType((16, ), "float32"))
        bn_mvar0 = relay.var("bn_var", relay.TensorType((16, ), "float32"))

        call1 = gv1(data, weight)
        call0 = gv0(call1, bn_gamma0, bn_beta0, bn_mmean0, bn_mvar0)
        mod["main"] = relay.Function([data, weight, bn_gamma0, bn_beta0, bn_mmean0,
                                      bn_mvar0], call0)
        mod = transform.InferType()(mod)
        return mod

    partitioned = partition()
    ref_mod = expected()

    assert tvm.ir.structural_equal(partitioned, ref_mod, map_free_vars=True)


if __name__ == "__main__":
    test_annotate()
    test_add()
    test_relu()
    test_conv2d()
    test_global_avg_pool2d()
