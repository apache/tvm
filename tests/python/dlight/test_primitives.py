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
from tvm import relay


@tvm.testing.requires_cuda
def test_normalize_primfunc_with_scalar():
    relay_text = """
    #[version = "0.0.5"]
    def @main(%p0: int32, Primitive=1) -> Tensor[(3), int32] {
    %0 = (20, %p0, 16);
    stack(%0)
    }
    """
    relay_graph = relay.fromtext(relay_text)

    f = tvm.get_global_func("relay.backend.LowerToPrimFunc")
    prim_func= f(relay_graph["main"],tvm.target.Target("cuda"))
    sch = tvm.tir.Schedule(prim_func)
    f_normalize_prim_func = tvm.get_global_func("tir.schedule.NormalizePrimFunc")
    assert f_normalize_prim_func(sch)

if __name__ == "__main__":
    tvm.testing.main()