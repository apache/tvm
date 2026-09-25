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
import tvm
import tvm.testing
from tvm.script import tirx as T


def test_texture_scope():
    @T.prim_func
    def texture_kernel(
        a: T.handle("float32", "global.texture"),
        c: T.handle("float32", "global.texture"),
    ):
        T.func_attr({"global_symbol": "texture_kernel", "calling_conv": 2})
        tx = T.env_thread("threadIdx.x")
        T.launch_thread(tx, 128)
        value = T.call_intrin("float32x4", "tirx.texture2d_load", a, tx, 0, 0, 128, T.Ramp(0, 1, 4))
        T.evaluate(
            T.call_intrin(
                "handle",
                "tirx.texture2d_store",
                c,
                tx,
                0,
                0,
                128,
                (value + T.Broadcast(T.float32(1), 4)) * T.Broadcast(T.float32(2), 4),
            )
        )

    target = tvm.target.Target({"kind": "opencl", "keys": ["adreno"]})
    mod = tvm.IRModule.from_expr(texture_kernel.with_attr("target", target))
    source = tvm.get_global_func("target.build.opencl")(mod, target).inspect_source()
    assert "__read_only image2d_array_t" in source
    assert "__write_only image2d_array_t" in source
    assert "READ_IMAGEF" in source
    assert "write_imagef" in source


if __name__ == "__main__":
    tvm.testing.main()
