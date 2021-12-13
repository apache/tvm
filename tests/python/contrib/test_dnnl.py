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
import mxnet as mx
from gluoncv.model_zoo import get_model

import numpy as np
import pytest
import itertools

import tvm
import tvm.relay.testing
from tvm import relay
from tvm.relay.op.contrib import dnnl
import tvm.testing
import argparse

has_dnnl_codegen = pytest.mark.skipif(
    not tvm.get_global_func("relay.ext.dnnl", True), reason="DNNL codegen not available"
)

run_module = tvm.testing.parameter(
    pytest.param(False, marks=[has_dnnl_codegen, *tvm.testing.requires_llvm()]),
    pytest.param(
        True, marks=[has_dnnl_codegen, *tvm.testing.requires_llvm()]
    ),
    ids=["compile", "run"],
)


def vmobj_to_list(o):
    if isinstance(o, tvm.nd.NDArray):
        return [o.numpy()]
    elif isinstance(o, tvm.runtime.container.ADT) or isinstance(o, list):
        return [vmobj_to_list(f) for f in o]
    else:
        raise RuntimeError("Unknown object type: %s" % type(o))


def assert_result_dict_holds(result_dict):
    for k1, k2 in itertools.combinations(result_dict, 2):
        res1 = vmobj_to_list(result_dict[k1])
        res2 = vmobj_to_list(result_dict[k2])
        for r1, r2 in zip(res1, res2):
            tvm.testing.assert_allclose(r1, r2, rtol=1e-3, atol=1e-3)


def run_and_verify_func(config, target="llvm", run_module=True):
    """Test a Relay func by compiling, running, and comparing TVM and DNNL outputs.

    Parameters
    ----------
    config : Tuple[relay.Function, Dict[str, NDArray], List[str]]
        A tuple containing 1) The function to test, 2) A dictionary of var names to input shapes and
        3) A list of which vars should be considered params.

    run_module: bool
        If True, the built module will be run after being compiled.
    """
    f, input_shapes, is_param = config
    params = {x: np.random.uniform(-1, 1, input_shapes[x]).astype(np.float32) for x in is_param}
    input_dict = {
        k: np.random.uniform(-1, 1, v).astype(np.float32)
        for k, v in input_shapes.items()
        if k not in is_param
    }
    dev = tvm.device(target)

    result_dict = dict()
    for mode in ["graph", "vm"]:
        for use_dnnl in [False, True]:
            mod = tvm.IRModule()
            mod["main"] = f
            result_key = mode + ("_dnnl" if use_dnnl else "")
            if use_dnnl:
                mod = dnnl.partition_for_dnnl(mod, params)
                with tvm.transform.PassContext(opt_level=3):
                    func = relay.create_executor(
                        mode, mod=mod, device=dev, target=target
                    ).evaluate()
            else:
                with tvm.transform.PassContext(opt_level=3):
                    func = relay.create_executor(
                        mode, mod=mod, device=dev, target=target
                    ).evaluate()
            if run_module:
                result_dict[result_key] = func(**input_dict, **params)

    if run_module:
        assert_result_dict_holds(result_dict)


def test_dnnl_not_compatible(run_module):
    dtype = "float32"
    xshape = (1, 32, 14, 14)
    x_data = np.random.uniform(-1, 1, xshape).astype(dtype)

    x = relay.var("x", shape=(xshape), dtype=dtype)
    y = relay.add(x, x)
    z = relay.cast(relay.cast(y, "int32"), "float32")
    out = relay.nn.relu(z)
    f = relay.Function([x], out)
    mod = tvm.IRModule()
    mod["main"] = f
    mod = dnnl.partition_for_dnnl(mod)
    for mode in ["graph", "vm"]:
        with tvm.transform.PassContext(opt_level=3):
            func = relay.create_executor(
                mode, mod=mod, device=tvm.cpu(0), target="llvm"
            ).evaluate()
            if run_module:
                results = func(x_data)


def test_conv2d(run_module):
    def get_graph(
        x_shape=(1, 32, 8, 8),
        k_shape=(16, 32, 3, 3),
        groups=1,
        padding=(0, 0),
        strides=(1, 1),
        dilation=(1, 1),
        channels=None,
    ):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        kernel = relay.var("kernel", shape=(k_shape), dtype="float32")
        out = relay.nn.conv2d(
            x,
            kernel,
            kernel_size=k_shape[2:4],
            groups=groups,
            padding=padding,
            strides=strides,
            dilation=dilation,
            channels=channels,
        )
        f = relay.Function([x, kernel], out)
        return f, {"x": x_shape, "kernel": k_shape}, ["kernel"]

    for k_shape, groups in [((16, 32, 3, 3), 1), ((32, 1, 3, 3), 32)]:
        for padding in [(0, 0), (1, 1)]:
            for strides in [(1, 1), (2, 2)]:
                for dilation in [(1, 1)]:
                    run_and_verify_func(
                        get_graph(
                            k_shape=k_shape,
                            groups=groups,
                            padding=padding,
                            strides=strides,
                            dilation=dilation,
                        ),
                        run_module=run_module,
                    )


def test_conv2d_weights_const(run_module):
    def get_graph(
        x_shape=(1, 32, 8, 8),
        k_shape=(16, 32, 3, 3),
        groups=1,
        padding=(0, 0),
        strides=(1, 1),
        dilation=(1, 1),
    ):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        kernel = relay.const(np.ones(k_shape).astype("float32"))
        out = relay.nn.conv2d(
            x,
            kernel,
            channels=k_shape[0],
            kernel_size=k_shape[2:4],
            groups=groups,
            padding=padding,
            strides=strides,
            dilation=dilation,
        )
        f = relay.Function([x], out)
        return f, {"x": x_shape}, []

    run_and_verify_func(get_graph(), run_module=run_module)


def test_dense(run_module):
    def get_graph(x_shape=(1, 16), k_shape=(32, 16)):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        kernel = relay.var("kernel", shape=(k_shape), dtype="float32")
        out = relay.nn.dense(x, kernel, units=k_shape[0])
        f = relay.Function([x, kernel], out)
        return f, {"x": x_shape, "kernel": k_shape}, ["kernel"]

    run_and_verify_func(get_graph(), run_module=run_module)
    run_and_verify_func(get_graph(k_shape=(1, 16)), run_module=run_module)


def test_multiple_outputs(run_module):
    def get_graph():
        x = relay.var("x", shape=(1, 3), dtype="float32")
        y = relay.var("y", shape=(1, 3), dtype="float32")
        z = relay.add(x, y)
        w = relay.add(z, y)
        out = relay.Tuple((z, w))
        f = relay.Function([x, y], out)
        return f, {"x": (1, 3), "y": (1, 3)}, []

    run_and_verify_func(get_graph(), run_module=run_module)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__] + sys.argv[1:]))