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
# pylint: disable=missing-docstring

import tvm
import tvm.testing
from tvm.script import tir as T


@T.prim_func
def main(p0: T.Buffer((), "int32"), T_stack: T.Buffer((T.int64(3),), "int32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    compile_engine_const = T.alloc_buffer((), "int32")
    compile_engine_const_1 = T.alloc_buffer((), "int32")
    with T.block("compile_engine_const"):
        vi = T.axis.spatial(1, T.int64(0))
        T.reads()
        T.writes(compile_engine_const[()])
        compile_engine_const[()] = 16
    with T.block("compile_engine_const_1"):
        vi = T.axis.spatial(1, T.int64(0))
        T.reads()
        T.writes(compile_engine_const_1[()])
        compile_engine_const_1[()] = 20
    for ax0 in range(T.int64(3)):
        with T.block("T_stack"):
            v_ax0 = T.axis.spatial(T.int64(3), ax0)
            T.reads(compile_engine_const[()], p0[()], compile_engine_const_1[()])
            T.writes(T_stack[v_ax0])
            T_stack[v_ax0] = T.if_then_else(
                v_ax0 == T.int64(2),
                compile_engine_const[()],
                T.if_then_else(v_ax0 == T.int64(1), p0[()], compile_engine_const_1[()]),
            )


@tvm.testing.requires_cuda
def test_normalize_primfunc_with_scalar():
    sch = tvm.tir.Schedule(main)
    f_normalize_prim_func = tvm.get_global_func("tir.schedule.NormalizePrimFunc")
    assert f_normalize_prim_func(sch)


if __name__ == "__main__":
    tvm.testing.main()
