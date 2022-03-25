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
"""Meta schedule integration with high-level IR"""
from typing import Dict, List, Optional, Union

import numpy as np  # type: ignore
import tvm.runtime.ndarray as nd
from tvm._ffi import get_global_func, register_object
from tvm.ir import IRModule, transform
from tvm.relay import Any
from tvm.relay import Function as RelayFunc
from tvm.runtime import NDArray, Object
from tvm.target import Target

from . import _ffi_api
from .database import Database
from .utils import autotvm_silencer


@register_object("meta_schedule.ExtractedTask")
class ExtractedTask(Object):
    """A tuning task extracted from the high-level IR

    Parameters
    ----------
    task_name : str
        The name of the task extracted
    mod : IRModule
        The high-level IR
    target: Target
        Target information
    dispatched : List[IRModule]
        A list of low-level IRs that the high-level IR could potentially dispatch to
    """

    task_name: str
    mod: IRModule
    dispatched: List[IRModule]

    def __init__(
        self,
        task_name: str,
        mod: IRModule,
        target: Target,
        dispatched: List[IRModule],
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.ExtractedTask,  # type: ignore # pylint: disable=no-member
            task_name,
            mod,
            target,
            dispatched,
        )


@register_object("meta_schedule.MetaScheduleContext")
class MetaScheduleContext(Object):
    """A context manager interface for the integration"""

    def query(
        self,
        task_name: str,
        mod: IRModule,
        target: Target,
        dispatched: Optional[List[IRModule]],
    ) -> Union[IRModule, None]:
        """The entry point of the integration

        Parameters
        ----------
        task_name : str
            The name of the task extracted
        mod : IRModule
            The high-level IR
        target: Target
            Target Info
        dispatched : Optional[List[IRModule]]
            A list of low-level IRs that the high-level IR could potentially dispatch to

        Returns
        -------
        result : IRModule or None
            Currently we only have to return tir::PrimFunc, but we wrap it under IRModule for
            more general future use. None is returned if there is no feedback hint.
        """
        return _ffi_api.MetaScheduleContextQuery(  # type: ignore # pylint: disable=no-member
            self,
            task_name,
            mod,
            target,
            dispatched,
        )

    @staticmethod
    def current() -> Optional["MetaScheduleContext"]:
        """The context manager in the current scope

        Returns
        -------
        ctx : Optional[MetaScheduleContext]
            The MetaScheduleContext in the current scope.
            NullOpt if it's currently not under any MetaScheduleContext.
        """
        return _ffi_api.MetaScheduleContextCurrent()  # type: ignore # pylint: disable=no-member

    @staticmethod
    def query_inside_with_scope(
        task_name: str,
        mod: IRModule,
        target: Target,
        dispatched: Optional[List[IRModule]],
    ) -> Union[IRModule, None]:
        """The entry point of the integration workflow. The compilation process of the high-level
        IR should call this method for task extraction and for feedback hints

        Basically, this method is equivalent to:

        .. code-block:: python

            def query_inside_with_scope(task_name, mod, dispatched):
                ctx = MetaScheduleContext.current()
                assert ctx is not None
                mod = ctx.query(task_name, mod, target, dispatched)

        Parameters
        ----------
        task_name : str
            The name of the task
        mod : IRModule
            The high-level IR
        target: Target
            Target
        dispatched : Optional[List[IRModule]]
            A list of low-level IRs that the high-level IR could potentially dispatch to

        Returns
        -------
        result : IRModule or None
            Currently we only have to return tir::PrimFunc, but we wrap it under IRModule for
            more general future use. None is returned if there is no feedback hint.
        """
        return _ffi_api.MetaScheduleContextQueryInsideWithScope(  # type: ignore # pylint: disable=no-member
            task_name,
            mod,
            target,
            dispatched,
        )

    def __enter__(self) -> "MetaScheduleContext":
        """Entering the scope of the context manager"""
        _ffi_api.MetaScheduleContextEnterScope(self)  # type: ignore # pylint: disable=no-member
        return self

    def __exit__(self, ptype, value, trace) -> None:
        """Exiting the scope of the context manager"""
        _ffi_api.MetaScheduleContextExitScope(self)  # type: ignore # pylint: disable=no-member


@register_object("meta_schedule.ApplyHistoryBest")
class ApplyHistoryBest(MetaScheduleContext):
    """An integration context that allows application of historically best record from database"""

    database: Database
    """ The database to be queried from"""

    def __init__(self, database) -> None:
        self.__init_handle_by_constructor__(_ffi_api.ApplyHistoryBest, database)  # type: ignore # pylint: disable=no-member


def extract_task_from_relay(
    mod: Union[IRModule, RelayFunc],
    target: Target,
    params: Optional[Dict[str, NDArray]] = None,
    *,
    opt_level: int = 3,
    pass_config: Optional[Dict[str, Any]] = None,
    disabled_pass: Optional[List[str]] = None,
) -> List[ExtractedTask]:
    """Extract tuning tasks from a relay program.

    Parameters
    ----------
    mod : Union[tvm.IRModule, tvm.relay.Function]
        The module or function to tune
    target : tvm.target.Target
        The compilation target
    params : Optional[Dict[str, tvm.runtime.NDArray]]
        The associated parameters of the program
    opt_level : int
        The optimization level of the compiler
    pass_config : Optional[Dict[str, Any]]
        The pass config of the compiler
    disabled_pass : Optional[List[str]]
        The list of disabled passes of the compiler

    Returns
    -------
    tasks: List[ExtractedTask]
        The tasks extracted from this network
    """

    extract_task_func = get_global_func("relay.backend.MetaScheduleExtractTask")
    assert extract_task_func

    target = Target(target) if isinstance(target, str) else target

    relay_params = {}
    for name, param in params.items():
        if isinstance(param, np.ndarray):
            param = nd.array(param)
        relay_params[name] = param

    if disabled_pass is None:
        disabled_pass = []
    if pass_config is None:
        pass_config = {"relay.backend.use_meta_schedule": True}

    if isinstance(mod, RelayFunc):
        mod = IRModule.from_expr(mod)
    if not isinstance(target, Target):
        target = Target(target)

    with autotvm_silencer(), target, transform.PassContext(
        opt_level=opt_level,
        config=pass_config,
        disabled_pass=disabled_pass,
    ):
        tasks = extract_task_func(mod, target, relay_params)
        # Tasks are extracted via post order visit, return the reversed list.
        return list(reversed(tasks))
