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

# pylint: disable=invalid-name
"""The build utils in python."""
from typing import Union, Optional
import tvm
from tvm.tir import PrimFunc
from tvm.ir.module import IRModule
from tvm.target import Target


def build(
    mod: Union[PrimFunc, IRModule],
    target: Optional[Union[str, Target]] = None,
    pipeline: Optional[Union[str, tvm.transform.Pass]] = "default_tir",
):
    return tvm.tir.build(mod, target, pipeline)
