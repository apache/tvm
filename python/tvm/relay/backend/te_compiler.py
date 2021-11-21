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
# pylint: disable=len-as-condition,no-else-return,invalid-name
"""TE compiler engine (replacing legacy compile_engine)."""
from __future__ import absolute_import

import logging
import tvm
from tvm import te, autotvm
from tvm.ir.transform import PassContext
from tvm.runtime import Object
from tvm.support import libinfo
from tvm.target import Target
from ..backend.utils import mangle_module_name
from .. import function as _function
from .. import ty as _ty
from . import _backend

logger = logging.getLogger("te_compiler")
autotvm_logger = logging.getLogger("autotvm")

_first_warning = True


@tvm._ffi.register_object("relay.LoweredOutput")
class LoweredOutput(Object):
    """Lowered output"""

    def __init__(self, outputs, implement):
        self.__init_handle_by_constructor__(_backend._make_LoweredOutput, outputs, implement)


@tvm._ffi.register_object("relay.CCacheKey")
class CCacheKey(Object):
    """Key in the TE Compiler.

    Parameters
    ----------
    source_func : tvm.relay.Function
        The source function.

    target : tvm.Target
        The target we want to run the function on.
    """

    def __init__(self, source_func, target):
        self.__init_handle_by_constructor__(_backend._make_CCacheKey, source_func, target)


@tvm._ffi.register_object("relay.CCacheValue")
class CCacheValue(Object):
    """Value in the TE Compiler, including usage statistics."""


def _get_cache_key(source_func, target):
    if isinstance(source_func, _function.Function):
        if isinstance(target, str):
            target = Target(target)
            if not target:
                raise ValueError("Need target when source_func is a Function")
        return CCacheKey(source_func, target)
    if not isinstance(source_func, CCacheKey):
        raise TypeError("Expect source_func to be CCacheKey")
    return source_func


def get_valid_implementations(op, attrs, inputs, out_type, target):
    """Get all valid implementations from the op strategy.

    Note that this function doesn't support op with symbolic input shapes.

    Parameters
    ----------
    op : tvm.ir.Op
        Relay operator.

    attrs : object
        The op attribute.

    inputs : List[tvm.te.Tensor]
        Input tensors to the op.

    out_type : relay.Type
        The output type.

    target : tvm.target.Target
        The target to compile the op.

    Returns
    -------
    ret : List[relay.op.OpImplementation]
        The list of all valid op implementations.
    """
    fstrategy = op.get_attr("FTVMStrategy")
    assert fstrategy is not None, (
        "%s doesn't have an FTVMStrategy registered. You can register "
        "one in python with `tvm.relay.op.register_strategy`." % op.name
    )
    with target:
        strategy = fstrategy(attrs, inputs, out_type, target)
    analyzer = tvm.arith.Analyzer()
    ret = []
    for spec in strategy.specializations:
        if spec.condition:
            # check if all the clauses in the specialized condition are true
            flag = True
            for clause in spec.condition.clauses:
                clause = analyzer.canonical_simplify(clause)
                if isinstance(clause, tvm.tir.IntImm) and clause.value:
                    continue
                flag = False
                break
            if flag:
                for impl in spec.implementations:
                    ret.append(impl)
        else:
            for impl in spec.implementations:
                ret.append(impl)
    return ret


def select_implementation(op, attrs, inputs, out_type, target, use_autotvm=True):
    """Select the best implementation from the op strategy.

    If use_autotvm is True, it'll first try to find the best implementation
    based on AutoTVM profile results. If no AutoTVM profile result is found,
    it'll choose the implementation with highest plevel.

    If use_autotvm is False, it'll directly choose the implementation with
    highest plevel.

    Note that this function doesn't support op with symbolic input shapes.

    Parameters
    ----------
    op : tvm.ir.Op
        Relay operator.

    attrs : object
        The op attribute.

    inputs : List[tvm.te.Tensor]
        Input tensors to the op.

    out_type : relay.Type
        The output type.

    target : tvm.target.Target
        The target to compile the op.

    use_autotvm : bool
        Whether query AutoTVM to pick the best.

    Returns
    -------
    ret : tuple(relay.op.OpImplementation, List[tvm.te.Tensor])
        The best op implementation and the corresponding output tensors.
    """
    all_impls = get_valid_implementations(op, attrs, inputs, out_type, target)
    best_plevel_impl = max(all_impls, key=lambda x: x.plevel)

    # Disable autotvm if auto_scheduler is enabled.
    # (i.e., always return the implementation with the highest priority for auto-scheduler).
    if PassContext.current().config.get("relay.backend.use_auto_scheduler", False):
        use_autotvm = False

    # If not use autotvm, always return the implementation with the highest priority
    if not use_autotvm:
        logger.info(
            "Using %s for %s based on highest priority (%d)",
            best_plevel_impl.name,
            op.name,
            best_plevel_impl.plevel,
        )
        outs = best_plevel_impl.compute(attrs, inputs, out_type)
        return best_plevel_impl, outs

    # Otherwise, try autotvm templates
    outputs = {}
    workloads = {}
    best_autotvm_impl = None
    best_cfg = None
    dispatch_ctx = autotvm.task.DispatchContext.current
    old_silent = autotvm.GLOBAL_SCOPE.silent
    autotvm.GLOBAL_SCOPE.silent = True
    for impl in all_impls:
        outs = impl.compute(attrs, inputs, out_type)
        outputs[impl] = outs
        workload = autotvm.task.get_workload(outs)
        workloads[impl] = workload
        if workload is None:
            # Not an AutoTVM tunable implementation
            continue
        cfg = dispatch_ctx.query(target, workload)
        if cfg.is_fallback:
            # Skip fallback config
            continue
        logger.info("Implementation %s for %s has cost %.2e", impl.name, op.name, cfg.cost)
        if best_cfg is None or best_cfg.cost > cfg.cost:
            best_autotvm_impl = impl
            best_cfg = cfg
    autotvm.GLOBAL_SCOPE.silent = old_silent

    if best_autotvm_impl:
        # The best autotvm implementation definitely doesn't use fallback config
        logger.info(
            "Using %s for %s based on lowest cost (%.2e)",
            best_autotvm_impl.name,
            op.name,
            best_cfg.cost,
        )
        return best_autotvm_impl, outputs[best_autotvm_impl]

    # Use the implementation with highest plevel
    if workloads[best_plevel_impl] is not None:
        msg = (
            "Cannot find tuning records for:\n    target=%s\n    key=%s\n"
            "TVM will apply a default schedule which may negatively impact performance."
            % (target, workloads[best_plevel_impl])
        )
        if (
            not autotvm.env.GLOBAL_SCOPE.silent
            and msg not in autotvm.task.DispatchContext.warning_messages
        ):
            autotvm.task.DispatchContext.warning_messages.add(msg)
            global _first_warning
            if _first_warning:
                _first_warning = False
                info_msg = (
                    "One or more operators have not been tuned. Please tune your model "
                    "for better performance. Use DEBUG logging level to see more details."
                )
                autotvm_logger.warning(info_msg)
            autotvm_logger.debug(msg)

    logger.info(
        "Using %s for %s based on highest priority (%s)",
        best_plevel_impl.name,
        op.name,
        best_plevel_impl.plevel,
    )
    return best_plevel_impl, outputs[best_plevel_impl]


def get_shape(shape):
    """Convert the shape to correct dtype and vars."""
    ret = []
    for dim in shape:
        if isinstance(dim, tvm.tir.IntImm):
            if libinfo()["INDEX_DEFAULT_I64"] == "ON":
                ret.append(dim)
            else:
                val = int(dim)
                assert val <= np.iinfo(np.int32).max
                ret.append(tvm.tir.IntImm("int32", val))
        elif isinstance(dim, tvm.tir.Any):
            ret.append(te.var("any_dim", "int32"))
        else:
            ret.append(dim)
    return ret


@tvm._ffi.register_func("relay.backend.lower_call")
def lower_call(call, inputs, target):
    """Lower the call expression to op implementation and tensor outputs."""
    assert isinstance(call.op, tvm.ir.Op)
    op = call.op

    # Prepare the call_node->checked_type(). For the call node inputs, we ensure that
    # the shape is Int32. Following code ensures the same for the output as well.
    # TODO(@icemelon9): Support recursive tuple
    ret_type = call.checked_type
    if isinstance(ret_type, _ty.TensorType):
        ret_type = _ty.TensorType(get_shape(ret_type.shape), ret_type.dtype)
    elif isinstance(ret_type, _ty.TupleType):
        new_fields = []
        for field in ret_type.fields:
            if isinstance(field, _ty.TensorType):
                new_fields.append(_ty.TensorType(get_shape(field.shape), field.dtype))
            else:
                new_fields.append(field)
        ret_type = _ty.TupleType(new_fields)

    is_dyn = _ty.is_dynamic(call.checked_type)
    for arg in call.args:
        is_dyn = is_dyn or _ty.is_dynamic(arg.checked_type)

    # check if in the AutoTVM tracing mode, and disable if op is not in wanted list
    env = autotvm.task.TaskExtractEnv.current
    reenable_tracing = False
    if env is not None and env.tracing:
        if env.wanted_relay_ops is not None and op not in env.wanted_relay_ops:
            env.tracing = False
            reenable_tracing = True

    if not is_dyn:
        best_impl, outputs = select_implementation(op, call.attrs, inputs, ret_type, target)
    else:
        # TODO(@icemelon9): Allow tvm to generate multiple kernels for dynamic shapes.
        best_impl, outputs = select_implementation(
            op, call.attrs, inputs, ret_type, target, use_autotvm=False
        )

    # re-enable AutoTVM tracing
    if reenable_tracing:
        env.tracing = True
    return LoweredOutput(outputs, best_impl)


@tvm._ffi.register_object("relay.TECompiler")
class TECompiler(Object):
    """TECompiler to get lowered code."""

    def __init__(self):
        raise RuntimeError("Cannot construct a TECompiler")

    def lower(self, source_func, target=None, mod_name="default"):
        """Lower a source_func to a CachedFunc.

        Parameters
        ----------
        source_func : Union[tvm.relay.Function, CCacheKey]
            The source relay function.

        target : tvm.Target
            The target platform.

        Returns
        -------
        cached_func: CachedFunc
            The result of lowering.
        """
        # pylint: disable=broad-except, import-outside-toplevel
        try:
            mod_name = mangle_module_name(mod_name)
            key = _get_cache_key(source_func, target)
            return _backend._TECompilerLower(self, key, mod_name)
        except Exception:
            import traceback

            msg = traceback.format_exc()
            msg += "Error during compile func\n"
            msg += "--------------------------\n"
            msg += source_func.astext(show_meta_data=False)
            msg += "--------------------------\n"
            raise RuntimeError(msg)

    def jit(self, source_func, target=None):
        """JIT a source_func to a tvm.runtime.PackedFunc.

        Parameters
        ----------
        source_func : Union[tvm.relay.Function, CCacheKey]
            The source relay function.

        target : tvm.Target
            The target platform.

        Returns
        -------
        jited_func: tvm.runtime.PackedFunc
            The result of jited function.
        """
        key = _get_cache_key(source_func, target)
        return _backend._TECompilerJIT(self, key)

    def clear(self):
        """clear the existing cached functions"""
        _backend._TECompilerClear(self)

    def items(self):
        """List items in the cache.
        Returns
        -------
        item_list : List[Tuple[CCacheKey, CCacheValue]]
            The list of items.
        """
        res = _backend._TECompilerListItems(self)
        assert len(res) % 2 == 0
        return [(res[2 * i], res[2 * i + 1]) for i in range(len(res) // 2)]


def get():
    """Get the global TE Compiler.

    Returns
    -------
    engine : tvm.relay.backend.TECompiler
        The TE Compiler.
    """
    return _backend._TECompilerGlobal()
