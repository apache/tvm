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
from tvm.relax.frontend.nn.llm._kernel_common import _get_prefill_kernel_config


def test_wide_head_prefill_uses_constrained_metal_tile():
    config = _get_prefill_kernel_config(
        h_kv=1,
        h_q=8,
        d=512,
        dtype="float16",
        target=tvm.target.Target("metal"),
    )

    _, _, _, _, num_warps, _, _, tile_z = config
    assert num_warps == 2
    assert tile_z == 8


if __name__ == "__main__":
    tvm.testing.main()
