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

import pytest

import tvm
import tvm.testing
from tvm import relay
from tvm.target.target import Target
from tvm.relay import testing
from tvm.relay.backend import Runtime, Executor, graph_executor_codegen


@pytest.mark.parametrize(
    "test_target,unsupported_config",
    [
        ["c", "-runtime=c"],
        ["c", "-system-lib=1"],
        ["c", "-executor=aot"],
        ["c", "-interface-api=c"],
        ["c", "-unpacked-api=1"],
        ["c", "-link-params=1"],
    ],
)
def test_deprecated_target_parameters(test_target, unsupported_config):
    with pytest.raises(ValueError) as e_info:
        Target(f"{test_target} {unsupported_config}")
        assert f"Cannot recognize '{unsupported_config}" in str(e_info.execption)


def test_build_relay_graph_():
    """Test to build a simple relay graph by using APIs directly"""

    def build_graph(mod, target):
        target, target_host = tvm.target.Target.canon_target_and_host(target)
        mod, _ = relay.optimize(mod, target)
        grc = graph_executor_codegen.GraphExecutorCodegen(None, target)
        _, lowered_funcs, _ = grc.codegen(mod, mod["main"])
        _ = relay.backend._backend.build(lowered_funcs, target)

    def add(shape, dtype):
        lhs = relay.var("A", shape=shape, dtype=dtype)
        rhs = relay.var("B", shape=shape, dtype=dtype)
        out = relay.add(lhs, rhs)
        expr = relay.Function((lhs, rhs), out)
        mod = tvm.IRModule.from_expr(expr)
        return mod

    build_graph(add((1, 8), "float32"), tvm.target.Target("llvm"))


@tvm.testing.requires_llvm
def test_schedule_record():
    """Test to build a nn model and get schedule_record from build_module"""

    def check_schedule(executor):
        for func_name, func_meta in executor.function_metadata.items():
            # check converted op only
            if "main" not in func_name:
                primfunc = list(func_meta.relay_primfuncs.values())[0]
                # make sure schedule is well-stored in function metadata
                assert "schedule" in primfunc.attrs
                sch = primfunc.attrs["schedule"]
                assert len(sch.schedule_record) == len(sch.primitive_record)

    relay_mod, params = testing.mobilenet.get_workload(batch_size=1, dtype="float32")
    target_llvm = tvm.target.Target("llvm")
    config = {"te.keep_schedule_record": True}

    with tvm.transform.PassContext(opt_level=3, config=config):
        aot_executor_factory = relay.build(
            relay_mod,
            target_llvm,
            runtime=Runtime("cpp"),
            executor=Executor("aot"),
            params=params,
        )
        graph_executor_factory = relay.build(
            relay_mod,
            target_llvm,
            params=params,
        )

    check_schedule(aot_executor_factory)
    check_schedule(graph_executor_factory)


if __name__ == "__main__":
    tvm.testing.main()
