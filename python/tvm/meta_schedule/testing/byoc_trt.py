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
"""TensorRT-MetaSchedule integration"""
# pylint: disable=import-outside-toplevel

from typing import List
import tvm
from tvm.runtime import Module
from tvm.meta_schedule.builder import BuilderResult
from tvm.target import Target


def relay_build_with_tensorrt(
    mod: Module,
    target: Target,
    params: dict,
) -> List[BuilderResult]:
    """Build a Relay IRModule with TensorRT BYOC
    Parameters
    ----------
    mod : IRModule
        The Relay IRModule to build.
    target : Target
        The target to build the module for.
    params : Dict[str, NDArray]
        The parameter dict to build the module with.
    Returns
    -------
    mod : runtime.Module
        The built module.
    """
    from tvm.relay.op.contrib.tensorrt import partition_for_tensorrt

    assert isinstance(target, Target)
    mod, config = partition_for_tensorrt(mod, params)
    with tvm.transform.PassContext(opt_level=3, config={"relay.ext.tensorrt.options": config}):
        result = tvm.relay.build_module._build_module_no_factory(mod, "cuda", "llvm", params)
    assert isinstance(result, Module)
    return result
