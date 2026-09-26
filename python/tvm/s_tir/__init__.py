# isort: skip_file
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
"""S-TIR namespace for scheduable TensorIR"""

import tvm.script

tvm.script.register_dialect("s_tir", "tvm.s_tir.script", builder_path="tvm.s_tir.script.ir_builder")

from ._tensor_intrin import TensorIntrin
from .stmt import MatchBufferRegion, SBlock, SBlockRealize

# dlight depends on compiler-only C++ functions (e.g. s_tir.schedule.GetSBlockRealize),
# so skip it in runtime-only builds.
from tvm.base import _RUNTIME_ONLY

from . import _ffi_api
from . import backend
from . import pipeline
from . import transform
from . import schedule
from .schedule import StmtSRef, SBlockScope, ScheduleState, Schedule, ScheduleError, Trace
from .sblock_dependence_info import SBlockDependenceInfo
from .data_layout import SLayout, SBijectiveLayout, sbijective_layout, slayout

if not _RUNTIME_ONLY:
    from . import analysis
    from . import meta_schedule
    from . import dlight


def renew_defs(func):
    """Re-generate the definition nodes for a TIR, including VarDef, BufferDef.
    This pass works as a simple DeepCopy to duplicate a function with different Vars and
    Buffers but the same behavior

    Parameters
    ----------
    func: PrimFunc
        The input function

    Returns
    -------
    result : PrimFunc
        The new generated func.
    """
    return _ffi_api.RenewDefs(func)


def _initialize_script_namespace():
    from . import script

    script._initialize()


from tvm.script.parser import register_namespace_initializer as _register_namespace_initializer

_register_namespace_initializer(_initialize_script_namespace, aliases=("s_tir",))


def _check_script_module(module):
    from tvm.tirx import PrimFunc

    if any(isinstance(fn, PrimFunc) and not fn.is_tirx for fn in module.functions.values()):
        from tvm.s_tir.script.ir_builder import _check_module_well_formed

        _check_module_well_formed(module)


tvm.script.register_module_validator(_check_script_module)
