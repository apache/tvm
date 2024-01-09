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
from tvm import te


def test_bound_tile_mod():
    def compute(M_tiles, N_tiles, factor, dtype):
        # Algo
        M = M_tiles * factor
        N = N_tiles * factor

        A = tvm.te.placeholder((N, M), name="A", dtype=dtype)
        C = tvm.te.compute((N, M), lambda n, m: A[n, m], name="C")
        s = tvm.te.create_schedule(C.op)

        return s, A, C

    def schedule(s, factor, padding, A, C):
        C_local = s.cache_write(C, "local")

        n, m = C.op.axis
        bn, bm, ni, mi = s[C].tile(n, m, factor, factor)
        nio, nii = s[C].split(ni, 2)
        n = s[C].fuse(nii, mi)
        C_shared = s.cache_write(C, "shared")
        bn, bm, ni, mi = C_shared.op.axis
        s[C_shared].storage_align(ni, factor * 2, padding)

        n, m = s[C].op.axis
        bn, bm, ni, mi = s[C].tile(n, m, factor, factor)
        s[C].set_scope("global")
        niio, niii = s[C].split(ni, 32)
        s[C_shared].compute_at(s[C], niio)

        return s

    s, A, C = compute(2, 2, 128, "float16")
    s = schedule(s, 128, 8, A, C)
    bounds = tvm.te.schedule.InferBound(s)
    check = bounds[s.stages[2].op.axis[2]].extent == 16
    if not check:
        print(tvm.lower(s, [A, C], simple_mode=True))
    assert check


if __name__ == "__main__":
    test_bound_tile_mod()
