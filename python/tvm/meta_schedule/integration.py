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
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Union

from tvm._ffi import register_object
from tvm.ir import IRModule, transform
from tvm.relay import Any, Function as RelayFunc, vm
from tvm.runtime import NDArray, Object
from tvm.target import Target
from tvm.tir import PrimFunc

from . import _ffi_api


@register_object("meta_schedule.ExtractedTask")
class ExtractedTask(Object):
    """A tuning task extracted from the high-level IR

    Parameters
    ----------
    task_name : str
        The name of the task extracted
    mod : IRModule
        The high-level IR
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
        dispatched: List[IRModule],
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.ExtractedTask,  # type: ignore # pylint: disable=no-member
            task_name,
            mod,
            dispatched,
        )


@register_object("meta_schedule.MetaScheduleContext")
class MetaScheduleContext(Object):
    """A context manager interface for the integration"""

    def query(
        self,
        task_name: str,
        mod: IRModule,
        dispatched: Optional[List[IRModule]],
    ) -> Union[IRModule, RelayFunc, PrimFunc, None]:
        """The entry point of the integration

        Parameters
        ----------
        task_name : str
            The name of the task extracted
        mod : IRModule
            The high-level IR
        dispatched : Optional[List[IRModule]]
            A list of low-level IRs that the high-level IR could potentially dispatch to

        Returns
        -------
        result : Union[IRModule, RelayFunc, PrimFunc, None]
            There are different types of the output:
            1) NullOpt if there is no feedback hint;
            2) tir::PrimFunc if `mod` should be lowered to a PrimFunc;
            3) relay::Function if `mod` should be dispatched to BYOC workflow;
            4) IRModule for unified dispatch
        """
        return _ffi_api.MetaScheduleContextQuery(  # type: ignore # pylint: disable=no-member
            self,
            task_name,
            mod,
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
        dispatched: Optional[List[IRModule]],
    ) -> Union[IRModule, RelayFunc, PrimFunc, None]:
        """The entry point of the integration workflow. The compilation process of the high-level
        IR should call this method for task extraction and for feedback hints

        Basically, this method is equivalent to:

        .. code-block:: python

            def query_inside_with_scope(task_name, mod, dispatched):
                ctx = MetaScheduleContext.current()
                assert ctx is not None
                ctx.query(task_name, mod, dispatched)

        Parameters
        ----------
        task_name : str
            The name of the task
        mod : IRModule
            The high-level IR
        dispatched : Optional[List[IRModule]]
            A list of low-level IRs that the high-level IR could potentially dispatch to

        Returns
        -------
        result : Union[IRModule, RelayFunc, PrimFunc, None]
            There are different types of the output:
            1) NullOpt if there is no feedback hint;
            2) tir::PrimFunc if `mod` should be lowered to a PrimFunc;
            3) relay::Function if `mod` should be dispatched to BYOC workflow;
            4) IRModule for unified dispatch
        """
        return _ffi_api.MetaScheduleContextQueryInsideWithScope(  # type: ignore # pylint: disable=no-member
            task_name,
            mod,
            dispatched,
        )

    def __enter__(self) -> "MetaScheduleContext":
        """Entering the scope of the context manager"""
        _ffi_api.MetaScheduleContextEnterScope(self)  # type: ignore # pylint: disable=no-member
        return self

    def __exit__(self, ptype, value, trace) -> None:
        """Exiting the scope of the context manager"""
        _ffi_api.MetaScheduleContextExitScope(self)  # type: ignore # pylint: disable=no-member


@register_object("meta_schedule.TaskExtraction")
class TaskExtraction(MetaScheduleContext):
    """An integration context for task extraction"""

    tasks: List[ExtractedTask]
    """The extracted tasks"""

    def __init__(self) -> None:
        self.__init_handle_by_constructor__(_ffi_api.TaskExtraction)  # type: ignore # pylint: disable=no-member


@register_object("meta_schedule.ApplyHistoryBest")
class ApplyHistoryBest(MetaScheduleContext):
    pass


def extract_task(
    mod: Union[IRModule, RelayFunc],
    target: Target,
    params: Optional[Dict[str, NDArray]] = None,
    *,
    opt_level: int = 3,
    pass_config: Dict[str, Any] = {
        "relay.backend.use_meta_schedule": True,
    },
    disabled_pass: List[str] = [],
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
    pass_config : Dict[str, Any]
        The pass config of the compiler
    disabled_pass : List[str]
        The list of disabled passes of the compiler

    Returns
    -------
    tasks: List[ExtractedTask]
        The tasks extracted from this network
    """

    @contextmanager
    def _autotvm_silencer():
        from tvm import autotvm  # pylint: disable=import-outside-toplevel

        silent = autotvm.GLOBAL_SCOPE.silent
        autotvm.GLOBAL_SCOPE.silent = True
        try:
            yield
        finally:
            autotvm.GLOBAL_SCOPE.silent = silent

    def _thread_run(func: Callable[[], None]) -> None:
        import threading  # pylint: disable=import-outside-toplevel

        thread = threading.Thread(target=func)
        thread.start()
        thread.join()

    env = TaskExtraction()
    if isinstance(mod, RelayFunc):
        mod = IRModule.from_expr(mod)
    if not isinstance(target, Target):
        target = Target(target)

    def _func():
        with env, _autotvm_silencer(), transform.PassContext(
            config=pass_config,
            disabled_pass=disabled_pass,
            opt_level=opt_level,
        ):
            compiler = vm.VMCompiler()
            if params:
                compiler.set_params(params)
            compiler.lower(mod, target)

    _thread_run(_func)
    return env.tasks
