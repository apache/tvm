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
"""Utility methods for the Universal Modular Accelerator Interface (UMA)"""

from enum import Enum, auto
import uuid
import tvm.tir
from tvm.contrib import utils, clang


class PassPhase(Enum):
    """UMA pass phases."""

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
