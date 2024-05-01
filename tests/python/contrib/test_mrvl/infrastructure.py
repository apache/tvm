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
# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name

"""Infrastructure to Test Marvell Code Generation"""
import json

import tvm
from tvm import relay
from tvm.relay.op.contrib import mrvl
import numpy as np
from tvm.contrib import graph_executor
from tvm.relay.build_module import build
from tvm.relay.op.contrib.mrvl import partition_for_mrvl


def get_cpu_op_count(mod):
    """Traverse graph counting ops offloaded to TVM."""

    class Counter(tvm.relay.ExprVisitor):
        def __init__(self):
            super().__init__()
            self.count = 0

        def visit_call(self, call):
            if isinstance(call.op, tvm.ir.Op):
                self.count += 1

            super().visit_call(call)

    c = Counter()
    c.visit(mod["main"])
    return c.count


def build_module(
    mod,
    target,
    params=None,
    enable_mrvl=True,
    tvm_ops=0,
    mrvl_partitions=1,
):
    """Partition and build module for mrvl codegen."""
    if isinstance(mod, tvm.relay.expr.Call):
        mod = tvm.IRModule.from_expr(mod)
    if params is None:
        params = {}

    with tvm.transform.PassContext(opt_level=3):
        if enable_mrvl:
            mod = mrvl.partition_for_mrvl(mod, params)
            tvm_op_count = get_cpu_op_count(mod)
            assert tvm_op_count == tvm_ops, "Got {} TVM operators, expected {}".format(
                tvm_op_count, tvm_ops
            )
            partition_count = 0
            for global_var in mod.get_global_vars():
                if "mrvl" in global_var.name_hint:
                    partition_count += 1

            assert mrvl_partitions == partition_count, "Got {} mrvl partitions, expected {}".format(
                partition_count, mrvl_partitions
            )
        return relay.build(mod, target, params=params)


def extract_mrvl_modules(module):
    """Get a list of all built mrvl runtime modules."""
    return list(filter(lambda mod: mod.type_key == "mrvl_sim", module.get_lib().imported_modules))


def verify_codegen(
    module, num_mrvl_modules=1, params=None, target="llvm", tvm_ops=0, contains=None
):
    """Check mrvl codegen against a known good output."""
    module = build_module(
        module,
        target,
        params=params,
        tvm_ops=tvm_ops,
        mrvl_partitions=num_mrvl_modules,
    )

    mrvl_modules = extract_mrvl_modules(module)
    assert len(mrvl_modules) == num_mrvl_modules, (
        f"The number of mrvl modules produced ({len(mrvl_modules)}) does not "
        f"match the expected value ({num_mrvl_modules})."
    )

    # Check if expected string is found inside actual string
    if contains is not None:
        actual_str = json.dumps(json.loads(mrvl_modules[0].get_source()))
        assert actual_str.find(contains)


def run_and_verify_func(config, data_type="float32"):

    np.random.seed(0)
    tvm_target = "llvm"

    func, input_shapes, is_param, option_dict = config
    params = {
        x: np.random.uniform(-1, 1, input_shapes[x]).astype(dtype=data_type) for x in is_param
    }
    inputs_dict = {
        k: np.random.uniform(-1, 1, v).astype(dtype=data_type)
        for k, v in input_shapes.items()
        if k not in is_param
    }

    dev = tvm.cpu()
    for use_mrvl in [True, False]:
        mod = tvm.IRModule()
        mod["main"] = func
        if use_mrvl:
            mod = partition_for_mrvl(mod, params, **option_dict)
            with tvm.transform.PassContext(
                opt_level=3, config={"relay.ext.mrvl.options": option_dict}
            ):
                model_lib = relay.build(mod, tvm_target, params=params)

            model_rt_graph = graph_executor.GraphModule(model_lib["default"](dev))
            model_rt_graph.set_input(**inputs_dict)
            model_rt_graph.run()
            output_tensor1 = model_rt_graph.get_output(0).numpy()

        else:
            with tvm.transform.PassContext(
                opt_level=3, config={"relay.ext.mrvl.options": option_dict}
            ):
                model_lib = relay.build(mod, tvm_target, params=params)

            model_rt_graph = graph_executor.GraphModule(model_lib["default"](dev))
            model_rt_graph.set_input(**inputs_dict)
            model_rt_graph.run()
            output_tensor2 = model_rt_graph.get_output(0).numpy()

    tvm.testing.assert_allclose(output_tensor1, output_tensor2, rtol=1e-2, atol=1e-2)
