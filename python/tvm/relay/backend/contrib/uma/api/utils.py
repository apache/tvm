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
"""Utility methods for the Universal Modular Accelerator Interface (UMA)"""

from enum import Enum, auto
import uuid

import tvm
import tvm.tir
from tvm.contrib import utils, clang


def uma_available() -> bool:
    registration_func = tvm.get_global_func(
        "relay.backend.contrib.uma.RegisterTarget", allow_missing=True
    )
    return registration_func is not None


class PassPhase(Enum):
    """
    UMA pass phases:

    PRE_PARTITIONING: prior to UMA partitioning
    POST_PARTITIONING_0: after UMA partitioning, before Defunctionalization
    POST_PARTITIONING_1: after UMA partitioning and after Defunctionalization
    TIR_PHASE_0: Generates the raw IR and loop levels.
    TIR_PHASE_1: Flattens the array storage.
    TIR_PHASE_2: Transforms loops, like unroll, vectorization and thread-binding.
    TIR_PHASE_3: Does some cleanup work.

    Reference to TIR phases: src/driver/driver_api.c
    """

    PRE_PARTITIONING = auto()
    POST_PARTITIONING_0 = auto()
    POST_PARTITIONING_1 = auto()
    TIR_PHASE_0 = auto()
    TIR_PHASE_1 = auto()
    TIR_PHASE_2 = auto()
    TIR_PHASE_3 = auto()


def _c_to_llvm(c_code: str) -> str:
    unique_filename = str(uuid.uuid4())
    temp = utils.tempdir()
    ll_path = temp.relpath(f"{unique_filename}.ll")
    ll_code = clang.create_llvm([c_code], output=ll_path)
    return ll_code


def add_llvm_to_block(
    sch: tvm.tir.Schedule, block_name: str, c_code_str: str = ""
) -> tvm.tir.Schedule:
    block = sch.get_block(block_name)
    loops = sch.get_loops(block)
    assert len(loops) > 0
    sch.annotate(loops[0], "pragma_import_llvm", _c_to_llvm(c_code_str))
    return sch
