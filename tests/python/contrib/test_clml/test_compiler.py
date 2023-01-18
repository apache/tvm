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
"""CLML compiler tests."""

import tvm
import numpy as np
from tvm import relay
from tvm.relay import testing
from tvm.relay.op.contrib import clml
import pytest


@tvm.testing.requires_openclml
def test_device_annotation():
    mod, params = relay.testing.mobilenet.get_workload(batch_size=1)
    mod = clml.partition_for_clml(mod, params)
    with tvm.transform.PassContext(opt_level=3):
        relay.backend.te_compiler.get().clear()
        lib = relay.build(
            mod,
            target="opencl -device=adreno",
            target_host="llvm -mtriple=aarch64-linux-gnu",
            params=params,
        )


if __name__ == "__main__":
    tvm.testing.main()
