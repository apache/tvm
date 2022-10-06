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

"""Add hexagon thread pool test"""

import numpy as np

import tvm
import tvm.contrib.hexagon
import tvm.script
import tvm.testing
from tvm.contrib.hexagon.session import Session
from tvm.script import tir as T

from .infrastructure import get_hexagon_target


@tvm.script.ir_module
class ElemwiseSumIRModule:
    """IRModule definition for elementwise sum"""

    # pylint: disable=no-self-argument,invalid-name,missing-function-docstring
    @T.prim_func
    def elemwise_sum_serial(a: T.handle, b: T.handle, c: T.handle, n: T.int32):
        T.func_attr({"global_symbol": "elemwise_sum_serial", "tir.noalias": True})
        A = T.match_buffer(a, (n,), dtype="float32")
        B = T.match_buffer(b, (n,), dtype="float32")
        C = T.match_buffer(c, (n,), dtype="float32")
        for i in T.serial(n):
            with T.block("C"):
                vi = T.axis.spatial(n, i)
                C[vi] = A[vi] + B[vi]

    @T.prim_func
    def elemwise_sum_parallel(a: T.handle, b: T.handle, c: T.handle, n: T.int32):
        T.func_attr({"global_symbol": "elemwise_sum_parallel", "tir.noalias": True})
        A = T.match_buffer(a, (n,), dtype="float32")
        B = T.match_buffer(b, (n,), dtype="float32")
        C = T.match_buffer(c, (n,), dtype="float32")
        for i in T.parallel(n):
            with T.block("C"):
                vi = T.axis.spatial(n, i)
                C[vi] = A[vi] + B[vi]

    # pylint: enable=no-self-argument,invalid-name,missing-function-docstring


def generate_add_test_data(hexagon_session: Session, n=128 * 1024):
    a = tvm.nd.array(np.random.uniform(size=n).astype("float32"), hexagon_session.device)
    b = tvm.nd.array(np.random.uniform(size=n).astype("float32"), hexagon_session.device)
    c = tvm.nd.array(np.zeros(n, dtype="float32"), hexagon_session.device)
    return (a, b, c, n)


def benchmark_func(mod, name, args, hexagon_session):
    (a, b, c, n) = args
    evaluator = mod.time_evaluator(name, hexagon_session.device, number=100)
    return evaluator(a, b, c, n).mean


@tvm.testing.requires_hexagon
def test_speedup(hexagon_session: Session, capsys):
    """Test speedup"""
    func = tvm.build(
        ElemwiseSumIRModule,
        target=get_hexagon_target("v68"),
    )
    mod = hexagon_session.load_module(func)
    args = generate_add_test_data(hexagon_session)
    parallel_mean = benchmark_func(mod, "elemwise_sum_parallel", args, hexagon_session)
    serial_mean = benchmark_func(mod, "elemwise_sum_serial", args, hexagon_session)

    with capsys.disabled():
        print("... speedup of {:.2f}".format(serial_mean / parallel_mean), end=" ")


@tvm.testing.requires_hexagon
def test_elemwise_sum_parallel(hexagon_session: Session):
    """Test parallel elementwise sum"""
    func = tvm.build(
        ElemwiseSumIRModule,
        target=get_hexagon_target("v68"),
    )
    mod = hexagon_session.load_module(func)

    (a, b, c, n) = generate_add_test_data(hexagon_session)
    mod["elemwise_sum_parallel"](a, b, c, n)
    tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())


if __name__ == "__main__":
    tvm.testing.main()
