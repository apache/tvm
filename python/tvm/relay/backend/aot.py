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
"""AOT passes"""
from typing import Dict

from tvm import IRModule
from tvm.relay.backend import Executor
from tvm.ir.transform import Pass
from .utils import CallType

from . import _aot


def AOTLowerMain(mod_name: str, config: object, call_type: CallType) -> Pass:
    """Lower a Relay main function into an AOT TIR main function.

    Parameters
    ----------
    mod_name: str
        The name of the module.
    config : CompilationConfig
        The compilation configuration.
    call_type : CallType
        The calling convention to use.

    Returns
    -------
    Pass
        The AOTLowerMain pass.

    """
    return _aot.AOTLowerMain(mod_name, config, call_type.value)


def CreateFunctionMetadata(
    mod: IRModule, workspace_byte_alignment: int, constant_byte_alignment: int
) -> Dict[str, object]:
    """Create the function metadata (FunctionInfos) from an AOT module.

    Parameters
    ----------
    mod : IRModule
        The IRModule.
    workspace_byte_alignment : int
        The alignment of the workspace buffer in bytes.
    constant_byte_alignment : int
        The alignment of the constant buffer in bytes.

    Returns
    -------
    Dict[str, FunctionInfo]
        A map between function names and FunctionInfos.

    """
    return _aot.CreateFunctionMetadata(mod, workspace_byte_alignment, constant_byte_alignment)


def CreateExecutorMetadata(
    mod: IRModule,
    mod_name: str,
    executor: Executor,
    workspace_byte_alignment: int,
    constant_byte_alignment: int,
) -> object:
    """Create the executor metadata from an AOT module.

    Parameters
    ----------
    mod : IRModule
        The IRModule.
    mod_name : str
        The name of the module.
    executor : Executor
        The executor configuration.
    workspace_byte_alignment : int
        The alignment of the workspace buffer in bytes.
    constant_byte_alignment : int
        The alignment of the constant buffer in bytes.

    Returns
    -------
    ExecutorCodegenMetadata
        The executor metadata.

    """
    return _aot.CreateExecutorMetadata(
        mod, mod_name, executor, workspace_byte_alignment, constant_byte_alignment
    )
