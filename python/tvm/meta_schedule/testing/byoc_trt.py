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
from tvm.runtime import Module
from tvm.target import Target
from tvm.meta_schedule.builder import BuilderInput, LocalBuilder, BuilderResult
from typing import List


def relay_build_with_tensorrt(
    mod: Module,
    target: Target,
    params: dict,
) -> List[BuilderResult]:
    from tvm.relay.op.contrib.tensorrt import partition_for_tensorrt

    mod, config = partition_for_tensorrt(mod, params)
    with tvm.transform.PassContext(opt_level=3, config={"relay.ext.tensorrt.options": config}):
        return tvm.relay.build_module._build_module_no_factory(mod, "cuda", "llvm", params)
