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
"""Target dependent BuildConfig for low-level passes."""
# TODO(tvm-team) consolidate with PassContext
import tvm._ffi
import tvm.ir

from tvm.runtime import Object
from tvm.tir import Stmt
from . import _ffi_api


class DumpIR(object):
    """
    Dump IR for each pass.
    With it, you can dump ir just like gcc/llvm.

    How to use:
    -----------
    .. code-block:: python

        with tvm.target.build_config(dump_pass_ir=True)
            run()
    """
    scope_level = 0
    def __init__(self):
        self._pass_id = 0
        self._recover_list = []

    def decorate(self, func):
        """ decorate the pass function"""
        def dump(*args, **kwargs):
            """dump function"""
            retv = func(*args, **kwargs)
            if not isinstance(retv, (Stmt,)):
                return retv
            fname = func.func_name if hasattr(func, 'func_name') else func.__name__
            pname = str(self._pass_id) + "_" + fname + "_ir.cc"
            with open(pname, "a") as f:
                out = retv
                f.write(str(out))
                self._pass_id += 1
            return retv
        return dump

    def decorate_irpass(self):
        """decorate ir_pass and ScheduleOps"""
        self._old_sgpass = schedule.ScheduleOps
        schedule.ScheduleOps = self.decorate(schedule.ScheduleOps)
        vset = vars(ir_pass)
        k = v = 0
        def recover():
            vset[k] = v
        for k, v in vset.items():
            self._recover_list.append(recover)
            vset[k] = self.decorate(v) if isinstance(v, Function) else v

    def decorate_custompass(self, custom_pass):
        """decorate given list of custom passes, and return decorated passes"""
        custom_pass = custom_pass if custom_pass else []
        pass_list = []
        for idx, x in enumerate(custom_pass):
            x[1].__name__ = "custom{}_phase{}".format(idx, x[0])
            pass_list += [(x[0], self.decorate(x[1]))]
        return pass_list

    def enter(self):
        """only decorate outermost nest"""
        if DumpIR.scope_level > 0:
            return
        self.decorate_irpass()
        self._pass_id = 0
        DumpIR.scope_level += 1

    def exit(self):
        """recover outermost nest"""
        if DumpIR.scope_level > 1:
            return
        # recover decorated functions
        for f in self._recover_list:
            f()
        schedule.ScheduleOps = self._old_sgpass
        DumpIR.scope_level -= 1


@tvm._ffi.register_object
class BuildConfig(Object):
    """Configuration scope to set a build config option.

    Note
    ----
    This object is backed by object protocol in C++, with arguments that can be
    exchanged between python and C++.

    Do not construct directly, use build_config instead.

    The fields that are backed by the C++ object are immutable once an instance
    is constructed. See _object_defaults for the fields.
    """

    _object_defaults = {
        "auto_unroll_max_step": 0,
        "auto_unroll_max_depth": 8,
        "auto_unroll_max_extent": 0,
        "unroll_explicit": True,
        "detect_global_barrier": False,
        "partition_const_loop": False,
        "offset_factor": 0,
        "data_alignment": -1,
        "restricted_func": True,
        "double_buffer_split_loop": 1,
        "dump_pass_ir": False,
        "instrument_bound_checkers": False,
        "disable_select_rewriting": False,
        "disable_vectorize": False,
        "disable_assert": False
    }
    _dump_ir = DumpIR()

    # pylint: disable=no-member
    def __init__(self, handle):
        """Initialize the function with handle

        Parameters
        ----------
        handle : SymbolHandle
            the handle to the underlying C++ Symbol
        """
        super(BuildConfig, self).__init__(handle)
        self.handle = handle

    @property
    def add_lower_pass(self):
        size = _ffi_api.BuildConfigGetAddLowerPassInfo(self)
        result = []
        for i in range(size):
            phase = _ffi_api.BuildConfigGetAddLowerPassInfo(self, i, True)
            func = _ffi_api.BuildConfigGetAddLowerPassInfo(self, i, False)
            result += [(phase, func)]
        return result

    @add_lower_pass.setter
    def add_lower_pass(self, value):
        add_lower_pass_args = []
        for x in value:
            add_lower_pass_args += [x[0], x[1]]
        _ffi_api.BuildConfigSetAddLowerPass(self, *add_lower_pass_args)

    def __enter__(self):
        # pylint: disable=protected-access
        _ffi_api.EnterBuildConfigScope(self)
        if self.dump_pass_ir:
            BuildConfig._dump_ir.enter()
        return self

    def __exit__(self, ptype, value, trace):
        if self.dump_pass_ir:
            BuildConfig._dump_ir.exit()
        _ffi_api.ExitBuildConfigScope(self)

    def __setattr__(self, name, value):
        if name in BuildConfig._object_defaults:
            raise AttributeError(
                "'%s' object cannot set attribute '%s'" % (str(type(self)), name))
        return super(BuildConfig, self).__setattr__(name, value)

    @staticmethod
    def current():
        """Get the current build configuration."""
        return _ffi_api.GetCurrentBuildConfig()


def build_config(**kwargs):
    """Configure the build behavior by setting config variables.

    Parameters
    ----------
    auto_unroll_max_step: int, default=0
        Threshold of number of steps in the loop to be automatically unrolled.
        This takes inner loop count into consideration.

    auto_unroll_max_depth: int, default=8
        The maximum nested level of loops that can be automatically unrolled.

    unroll_explicit: bool, default=True
        Whether explicitly unroll the loop, if set false, the unroll hint will
        be passed to the CodeGen phase, which may generate pragma unroll hint.
        Set this to be true if CodeGen support unroll pragma and
        when we want to be more readable.

    detect_global_barrier: bool, default=True
        Whether detect global barrier.

    partition_const_loop: bool, default=False
        Whether partition const loop

    data_alignment: int, optional
        The alignment of data pointer in bytes.
        If -1 is passed, the alignment will be set to TVM's internal default.

    offset_factor: int, default=0
        The factor used in default buffer declaration.
        If specified as 0, offset field is not used.

    restricted_func: bool, default=True
        Whether build restricted function.
        That is each buffer argument to the function are guaranteed
        not to overlap. This enables more optimization.
        Corresponds to restricted keyword in C99

    double_buffer_split_loop: int, default=2
        Whether split the loop with factor. If it is zero, no splitting will happen.
        It it is bigger than one, the logic will do a split with factor equals the integer
        and unroll the inner loop. This allows the buffer fetching won't contain condition.

    add_lower_pass: list of tuple (phase, function(Stmt->Stmt)), default=None
        phase contains an integer on which optimization pass we apply the pass.
        Additional lowering passes to be applied before make_api.

    dump_pass_ir: dump ir of each pass into file idx_passname_ir.cc, default=False

    Returns
    -------
    config: BuildConfig
        The build configuration
    """
    node_args = {k: v if k not in kwargs else kwargs[k]
                 for k, v in BuildConfig._object_defaults.items()}
    config = tvm.ir.make_node("BuildConfig", **node_args)

    if "add_lower_pass" in kwargs:
        config.add_lower_pass = kwargs["add_lower_pass"]

    return config
