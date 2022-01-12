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
"""Test DNNL integration conv2d tests."""

import numpy as np
import pytest
import tvm
from tvm import relay, runtime
from tvm.relay.backend import te_compiler
from tvm.contrib import graph_executor

import collections
from numbers import Number

requires_dnnl = pytest.mark.skipif(
    tvm.get_global_func("runtime.DNNLJSONRuntimeCreate", True) is None,
    reason="DNNL codegen is not available",
)


def parametrized(arg_name, workloads):
    return pytest.mark.parametrize(
        arg_name, [w[1:] for w in workloads], ids=[w[0] for w in workloads]
    )


def permute(shape, l_from="", l_to=""):
    res_shape = []
    for label in l_to:
        pos = l_from.find(label)
        res_shape.append(shape[pos])

    return res_shape


def expand_dim(shape, rank=0):
    assert len(shape) == 1
    return shape + [1] * (rank - 1)


def check_fully_annotated(mod, desired_compiler):
    matched_ops = []
    other_ops = []

    def _visit(node):
        if isinstance(node, tvm.relay.Call):
            op = node.op
            if isinstance(op, relay.GlobalVar):
                func = mod[op]
                if "Compiler" in func.attrs and func.attrs["Compiler"] == desired_compiler:
                    matched_ops.append(op)
                    return
            else:
                other_ops.append(op)

    tvm.relay.analysis.post_order_visit(mod["main"].body, _visit)

    assert len(other_ops) == 0 and len(matched_ops) != 0, "Model is not fully DNNL compiled"


def check_result(
    mod,
    ref_mod,
    map_inputs,
    tol=1e-5,
    target="llvm",
    device=tvm.cpu(),
    params=None,
    ref_result=None,
    atol=None,
    desired_compiler="dnnl"
):
    if atol is None:
        atol = tol

    if desired_compiler is not None:
        check_fully_annotated(mod, desired_compiler)

    if ref_result is None:
        # Run the reference result
        te_compiler.get().clear()
        with tvm.transform.PassContext(opt_level=3):
            ref_lib = relay.build(ref_mod, target=target, params=params)
        ref_rt_mod = tvm.contrib.graph_executor.GraphModule(ref_lib["default"](device))

        for name, data in map_inputs.items():
            ref_rt_mod.set_input(name, data)
        ref_rt_mod.run()
        out = ref_rt_mod.get_output(0)
        ref_result = out.numpy()

    def check_vm_result():
        te_compiler.get().clear()
        with tvm.transform.PassContext(opt_level=3):
            exe = relay.vm.compile(mod, target=target, params=params)
        code, lib = exe.save()
        exe = runtime.vm.Executable.load_exec(code, lib)
        vm = runtime.vm.VirtualMachine(exe, device)
        output = vm.run(**map_inputs)
        tvm.testing.assert_allclose(output.numpy(), ref_result, rtol=tol, atol=atol)

    def check_graph_executor_result():
        te_compiler.get().clear()
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)
        rt_mod = tvm.contrib.graph_executor.GraphModule(lib["default"](device))

        rt_mod.run(**map_inputs)
        output = rt_mod.get_output(0)
        tvm.testing.assert_allclose(output.numpy(), ref_result, rtol=tol, atol=atol)

    check_vm_result()
    check_graph_executor_result()


def filler_uni(low=0, high=1):
    def filler_func(shape):
        return np.random.uniform(low, high, shape)

    return filler_func


class Builder:
    def __init__(self, qnn_profile=None):
        self._args = {}
        self._args_op = []
        self._qp = qnn_profile

    def arg(self, shape=[], dtype="float32", filler=filler_uni(), is_const=True):
        if isinstance(filler, Number):
            value = np.full(shape, filler).astype(dtype)
        else:
            value = filler(shape).astype(dtype)

        if is_const:
            res = relay.const(value, dtype=dtype)
        else:
            name = f"in_{len(self._args)}"
            res = relay.var(name, shape=shape, dtype=dtype)
            self._args[name] = value
            self._args_op.append(res)

        return res

    def make_zp(self, mean_val, num_ch=1, dispersion=0.2):
        if num_ch == 1:
            return self.arg(shape=[], dtype="int32", filler=mean_val)
        else:
            low = int(mean_val * (1 - dispersion))
            high = int(mean_val * (1 + dispersion))
            return self.arg(shape=[num_ch], dtype="int32", filler=filler_uni(low, high))

    def make_scl(self, mean_val, num_ch=1, dispersion=0.2):
        if num_ch == 1:
            return self.arg(shape=[], dtype="float32", filler=mean_val)
        else:
            low = mean_val * (1 - dispersion)
            high = mean_val * (1 + dispersion)
            return self.arg(shape=[num_ch], dtype="float32", filler=filler_uni(low, high))

    def make_zp_and_scl(self, name, num_ch=1, dispersion=0.2):
        is_per_channel = getattr(self._qp, f"{name}_pc")
        zp_val = getattr(self._qp, f"{name}_zp")
        scl_val = getattr(self._qp, f"{name}_scl")

        zp = self.make_zp(zp_val, num_ch if is_per_channel else 1, dispersion)
        scl = self.make_scl(scl_val, num_ch if is_per_channel else 1, dispersion)
        return zp, scl

    def finalize(self, op):
        func = relay.Function(self._args_op, op)
        mod = tvm.IRModule.from_expr(func)
        mod = relay.transform.InferType()(mod)
        return mod, self._args


ConvProfile = collections.namedtuple(
    "ConvProfile",
    [
        "N",
        "IH",
        "IW",
        "IC",
        "OC",
        "KH",
        "KW",
        "SH",
        "SW",
        "PH",
        "PW",
        "DH",
        "DW",
        "GR",
        "D_LAYOUT",
        "K_LAYOUT",
    ],
)

DenseProfile = collections.namedtuple("DenseProfile", ["N", "IC", "OC"])

ArgConstConfig = collections.namedtuple("ArgConstConfig", ["Data", "Weights", "Bias", "Sum"])

QuantizationConfig = collections.namedtuple(
    "QuantizationConfig",
    [
        "d_zp",
        "d_scl",
        "d_pc",
        "k_zp",
        "k_scl",
        "k_pc",
        "rq_zp",
        "rq_scl",
        "rq_pc",
        "sum_zp",
        "sum_scl",
        "sum_pc",
        "o_zp",
        "o_scl",
        "o_pc",
    ],
)
