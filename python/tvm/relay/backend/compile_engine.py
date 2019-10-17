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
# pylint: disable=len-as-condition,no-else-return
"""Backend code generation engine."""
from __future__ import absolute_import

import hashlib
import numpy as np
import tvm
from topi import tag
from ..base import register_relay_node, Object
from ... import _api_internal
from ... import target as _target
from ..._ffi.function import register_func
from ... import autotvm
from .. import expr as _expr
from .. import op as _op
from .. import ty as _ty
from ..expr_functor import ExprVisitor
from . import _backend

@register_relay_node
class CachedFunc(Object):
    """Low-level tensor function to back a relay primitive function.
    """
    def __init__(self, target, func_name, inputs, outputs, schedule=None,
                 lowered_funcs=None, shape_func_param_states=None):
        if lowered_funcs is None:
            lowered_funcs = []
        if shape_func_param_states is None:
            shape_func_param_states = []
        self.__init_handle_by_constructor__(
            _backend._make_CachedFunc, target, func_name, inputs, outputs,
            schedule, lowered_funcs, shape_func_param_states)


@register_relay_node
class CCacheKey(Object):
    """Key in the CompileEngine.

    Parameters
    ----------
    source_func : tvm.relay.Function
        The source function.

    target : tvm.Target
        The target we want to run the function on.
    """
    def __init__(self, source_func, target):
        self.__init_handle_by_constructor__(
            _backend._make_CCacheKey, source_func, target)


@register_relay_node
class CCacheValue(Object):
    """Value in the CompileEngine, including usage statistics.
    """


def _get_cache_key(source_func, target):
    if isinstance(source_func, _expr.Function):
        if isinstance(target, str):
            target = _target.create(target)
            if not target:
                raise ValueError("Need target when source_func is a Function")
        return CCacheKey(source_func, target)
    if not isinstance(source_func, CCacheKey):
        raise TypeError("Expect source_func to be CCacheKey")
    return source_func


def get_shape(shape):
    """Convert the shape to correct dtype and vars."""
    ret = []
    for dim in shape:
        if isinstance(dim, tvm.expr.IntImm):
            val = int(dim)
            assert val <= np.iinfo(np.int32).max
            ret.append(tvm.expr.IntImm("int32", val))
        elif isinstance(dim, tvm.expr.Any):
            ret.append(tvm.var("any_dim", "int32"))
        else:
            ret.append(dim)
    return ret


def get_valid_implements(op, attrs, inputs, out_type, target):
    """Get all valid implementations from the op strategy.

    Note that this function doesn't support op that has symbolic input shapes.

    Parameters
    ----------
    op : relay.op.Op
        Relay operator.

    attrs : object
        The op attribute.

    inputs : list of tvm.Tensor
        Input tensors to the op.

    out_type : relay.Type
        The output type.

    target : tvm.Target
        The target to compile the op.

    Returns
    -------
    ret : list of relay.op.OpImplement
        The list of op implementations.
    """
    fstrategy = op.get_attr("FTVMStrategy")
    assert fstrategy is not None, "%s doesn't have FTVMStrategy registered" % op.name
    with target:
        strategy = fstrategy(attrs, inputs, out_type, target)
    ret = []
    for spec in strategy.specializations:
        if spec.condition:
            # check if all the clauses in the specialized condition are true
            flag = True
            for clause in spec.condition.clauses:
                clause = tvm.ir_pass.Simplify(clause)
                if isinstance(clause, tvm.expr.IntImm) and clause.value:
                    continue
                flag = False
                break
            if flag:
                for impl in spec.implements:
                    ret.append(impl)
        else:
            for impl in spec.implements:
                ret.append(impl)
    return ret


def select_implement(op, attrs, inputs, out_type, target, use_autotvm=True):
    """Select the best implement from the op strategy.

    If use_autotvm is True, it'll first try to find the best implementation
    based on AutoTVM profile results. If no AutoTVM profile result is found,
    it'll choose the implementation with highest plevel.

    If use_autotvm is False, it'll directly choose the implementation with
    highest plevel.

    Note that this function doesn't support op that has symbolic input shapes.

    Parameters
    ----------
    op : relay.op.Op
        Relay operator.

    attrs : object
        The op attribute.

    inputs : list[tvm.Tensor]
        Input tensors to the op.

    out_type : relay.Type
        The output type.

    target : tvm.Target
        The target to compile the op.

    use_autotvm : bool
        Whether query AutoTVM to pick the best.

    Returns
    -------
    ret : tuple(relay.op.OpImplement, list[tvm.Tensor])
        The best op implementation and the corresponding output tensors.
    """
    all_impls = get_valid_implements(op, attrs, inputs, out_type, target)

    best_plevel_impl = None
    for impl in all_impls:
        if best_plevel_impl is None or int(impl.plevel) > int(best_plevel_impl.plevel):
            best_plevel_impl = impl
    if not use_autotvm:
        outs = best_plevel_impl.compute(attrs, inputs, out_type)
        return best_plevel_impl, outs

    outputs = {}
    best_autotvm_impl = None
    best_cfg = None
    dispatch_ctx = autotvm.task.DispatchContext.current
    for impl in all_impls:
        outs = impl.compute(attrs, inputs, out_type)
        outputs[impl] = outs
        workload = autotvm.task.get_workload(outs)
        if workload is None:
            continue
        cfg = dispatch_ctx.query(target, workload)
        if cfg.cost is None:
            # It's a fallback config
            continue
        if best_cfg is None or best_cfg.cost > cfg.cost:
            best_autotvm_impl = impl
            best_cfg = cfg
    if best_autotvm_impl:
        return best_autotvm_impl, outputs[best_autotvm_impl]
    return best_plevel_impl, outputs[best_plevel_impl]


class ScheduleGetter(ExprVisitor):
    """Get the schedule given a fused Relay function"""

    MAX_FUNC_NAME_LENGTH = 80

    def __init__(self, target):
        super().__init__()
        self.target = target
        self.master_op = None
        self.master_attrs = None
        self.master_op_pattern = 0
        self.master_implement = None
        self.func_name = ""
        self.scalars = []
        self._device_copy_op = _op.get("device_copy")

    def create(self, prim_func):
        """Get the schedule and create the cached function"""
        assert isinstance(prim_func, _expr.Function)
        assert prim_func.is_primitive()

        def create_tensors(typ, tensors):
            if isinstance(typ, _ty.TensorType):
                tensors.append(tvm.placeholder(get_shape(typ.shape), typ.dtype))
            else:
                assert isinstance(typ, _ty.TupleType)
                for field in typ.fields:
                    create_tensors(field, tensors)

        inputs = []
        for param in prim_func.params:
            tensors = []
            create_tensors(param.checked_type, tensors)
            self.memo_map[param] = tensors
            inputs.extend(tensors)
        self.func_name = "fused"
        outputs = self.visit(prim_func.body)
        if len(self.func_name) > ScheduleGetter.MAX_FUNC_NAME_LENGTH:
            hash_digest = int(hashlib.sha1(self.func_name).hexdigest(), 16)
            self.func_name = "%s_%s" % (
                self.func_name[:ScheduleGetter.MAX_FUNC_NAME_LENGTH], hash_digest)

        assert self.master_op is not None
        tensor_outs = []
        for tensor in outputs:
            if not isinstance(tensor.op, tvm.tensor.PlaceholderOp):
                tensor_outs.append(tensor)
        sch = None
        if not isinstance(self.master_attrs, _op.op_attrs.DeviceCopyAttrs):
            # print('master op:', self.master_op.name)
            sch = self.master_implement.schedule(self.master_attrs, tensor_outs, self.target)
            for scalar in self.scalars:
                sch[scalar].compute_inline()
        return CachedFunc(self.target, self.func_name, inputs, outputs, sch)

    def visit_var(self, var):
        assert False, "Found free variable " + var.name_hint

    def visit_constant(self, const):
        assert len(const.data.shape) == 0, "Constant is not scalar"
        dtype = const.data.dtype
        data = const.data.asnumpy()
        def fcompute():
            if dtype.startswith("int"):
                return tvm.expr.IntImm(dtype, int(data))
            elif dtype.startswith("uint"):
                return tvm.expr.UIntImm(dtype, int(data))
            elif dtype.startswith("float"):
                return tvm.expr.FloatImm(dtype, float(data))
            else:
                assert False, "not handled"
                return tvm.expr.Expr()
        value = tvm.compute((), fcompute, name="compile_engine_const", tag=tag.BROADCAST)
        self.scalars.append(value.op)
        return [value]

    def visit_call(self, call):
        inputs = []
        count_tuple = 0
        for arg in call.args:
            if isinstance(arg.checked_type, _ty.TupleType):
                count_tuple += 1
            inputs.extend(self.visit(arg))
        assert count_tuple <= 1, "Only allow function with a single tuple input"
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
        assert isinstance(call.op, _op.Op)
        op = call.op

        # disable AutoTVM tracing if op is not in wanted list
        env = autotvm.task.TaskExtractEnv.current
        reenable_tracing = False
        if env is not None and env.tracing:
            if env.wanted_relay_ops is not None and op not in env.wanted_relay_ops:
                env.tracing = False
                reenable_tracing = True

        if op == self._device_copy_op:
            copy_input = inputs[0]
            outputs = [_api_internal._Tensor(copy_input.shape, copy_input.dtype,
                                             None, 0)]
        else:
            is_dyn = call.checked_type.is_dynamic()
            for arg in call.args:
                is_dyn = is_dyn or arg.checked_type.is_dynamic()

            if not is_dyn:
                best_impl, outputs = select_implement(
                    op, call.attrs, inputs, ret_type, self.target)
            else:
                # TODO(@icemelon9): Allow tvm to generate multiple kernels for dynamic shapes
                # for dynamic case, we currently use the implementation with highest plevel
                best_impl, outputs = select_implement(
                    op, call.attrs, inputs, ret_type, self.target, use_autotvm=False)
        op_pattern = op.get_attr("TOpPattern")
        if op_pattern >= _op.OpPattern.COMM_REDUCE:
            assert self.master_op is None or self.master_op_pattern < _op.OpPattern.COMM_REDUCE, \
                "Two complicated op in a primitive function master=%s current=%s" % (
                    self.master_op, op)
        if op_pattern >= self.master_op_pattern:
            self.master_op = op
            self.master_attrs = call.attrs
            self.master_op_pattern = op_pattern
            self.master_implement = best_impl
        if len(outputs) > 1:
            assert isinstance(call.checked_type, _ty.TupleType)
            assert len(call.checked_type.fields) == len(outputs)
        if op == self._device_copy_op:
            self.func_name += "__copy"
        else:
            self.func_name += "_" + op.name

        # re-enable AutoTVM tracing
        if reenable_tracing:
            env.tracing = True

        return outputs

    def visit_let(self, let):
        val = self.visit(let.value)
        assert let.var not in self.memo_map
        self.memo_map[let.var] = val
        return self.visit(let.body)

    def visit_tuple(self, tup):
        fields = []
        for field in tup.fields:
            assert isinstance(field.checked_type, _ty.TensorType), "Only allow Tuple of Tensor"
            res = self.visit(field)
            assert len(res) == 1
            fields.append(res[0])
        return fields

    def visit_tuple_getitem(self, t):
        tup = self.visit(t.tuple)
        assert len(tup) == len(t.tuple.checked_type.fields)
        assert t.index >= 0
        assert t.index < tup.size()
        return [tup[t.index]]


@register_func("relay.backend.create_schedule")
def create_schedule(src_func, target):
    return ScheduleGetter(target).create(src_func)


@register_relay_node
class CompileEngine(Object):
    """CompileEngine to get lowered code.
    """
    def __init__(self):
        raise RuntimeError("Cannot construct a CompileEngine")

    def lower(self, source_func, target=None):
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
            key = _get_cache_key(source_func, target)
            return _backend._CompileEngineLower(self, key)
        except Exception:
            import traceback
            msg = traceback.format_exc()
            msg += "Error during compile func\n"
            msg += "--------------------------\n"
            msg += source_func.astext(show_meta_data=False)
            msg += "--------------------------\n"
            raise RuntimeError(msg)

    def lower_shape_func(self, source_func, target=None):
        key = _get_cache_key(source_func, target)
        return _backend._CompileEngineLowerShapeFunc(self, key)

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
        return _backend._CompileEngineJIT(self, key)

    def clear(self):
        """clear the existing cached functions"""
        _backend._CompileEngineClear(self)

    def items(self):
        """List items in the cache.

        Returns
        -------
        item_list : List[Tuple[CCacheKey, CCacheValue]]
            The list of items.
        """
        res = _backend._CompileEngineListItems(self)
        assert len(res) % 2 == 0
        return [(res[2*i], res[2*i+1]) for i in range(len(res) // 2)]

    def dump(self):
        """Return a string representation of engine dump.

        Returns
        -------
        dump : str
            The dumped string representation
        """
        items = self.items()
        res = "====================================\n"
        res += "CompilerEngine dump, %d items cached\n" % len(items)
        for k, v in items:
            res += "------------------------------------\n"
            res += "target={}\n".format(k.target)
            res += "use_count={}\n".format(v.use_count)
            res += "func_name={}\n".format(v.cached_func.func_name)
            res += k.source_func.astext() + "\n"
        res += "===================================\n"
        return res


def get():
    """Get the global compile engine.

    Returns
    -------
    engine : tvm.relay.backend.CompileEngine
        The compile engine.
    """
    return _backend._CompileEngineGlobal()
