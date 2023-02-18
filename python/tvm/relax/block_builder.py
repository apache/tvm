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
# pylint: disable=no-else-return, invalid-name
"""Developer API of constructing Relax AST."""
import typing

from typing import Dict, List, Optional, Union, Any, Callable
from tvm.ir.module import IRModule
from tvm.runtime import Object
from tvm import relax as rx, tir
import tvm
from .expr import (
    Expr,
    te_tensor,
    Var,
    ShapeExpr,
    GlobalVar,
    BindingBlock,
    Tuple,
    BaseFunc,
    Binding,
)
from .struct_info import PrimStructInfo, ShapeStructInfo, StructInfo, TensorStructInfo
from .op.base import call_tir
from . import _ffi_api


class FunctionScope(object):
    """Auxiliary scope for function"""

    def __init__(self, block_builder, name, params, attrs):
        self._bb = block_builder
        self._name = name
        self._params = params
        self._attrs = attrs

    def __enter__(self):
        self._bb._enter_function_scope(self._name, self._params, self._attrs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # __exit__ should properly handle the case where the with block exits with an exception
        # when handling error case in exit, always check if there is already an exception
        # been thrown in the with block
        self._bb._exit_function_scope(exc_type, exc_val, exc_tb)


class DataflowScope(object):
    """Auxiliary scope for Dataflow block"""

    def __init__(self, block_builder):
        self._bb = block_builder

    def __enter__(self):
        block = self._bb._end_block()
        if len(block.bindings) > 0:
            self._bb._blocks.append(block)
        self._bb._begin_dataflow_block()

    def __exit__(self, ptype, value, trace):
        block = self._bb._end_block()
        if len(block.bindings) > 0:
            self._bb._blocks.append(block)
        self._bb._begin_binding_block()


class TestingScope(object):
    """Auxiliary scope for testing purposes"""

    def __init__(self, block_builder, def_vars):
        self._bb = block_builder
        shape_vars = []
        for var in def_vars:
            if isinstance(var, tvm.tir.Var):
                shape_vars.append(var)
            else:
                raise ValueError("def_vars only can take tir.Var")
        # setup a dummy var so shape is in scope.
        sparam = rx.Var("sparam", rx.ShapeStructInfo(shape_vars))
        self._scope_params = [sparam]

    def __enter__(self):
        self._bb.begin_scope(self._scope_params)
        self._bb._begin_dataflow_block()

    def __exit__(self, ptype, value, trace):
        self._bb._end_block()
        self._bb.end_scope()


@tvm._ffi.register_object("relax.BlockBuilder")
class BlockBuilder(Object):
    """A builder to build Relax IR for testing and dev.

    Examples
    --------
    .. code-block:: python

        m = tir.Var("m", "int32")
        n = tir.Var("n", "int32")
        x = rx.Var("x", rx.TensorStructInfo([m, n], "float16"))
        y = rx.Var("y", rx.TensorStructInfo([n], "float16")
        bb = rx.BlockBuilder()
        with bb.function([x, y], "func"):
            with bb.dataflow() as df:
                lv0 = bb.emit(rx.add(x, y))
                lv1 = bb.emit(rx.multiply(lv0, y))
                gv0 = bb.emit_output(lv1)
            bb.emit_func_output(gv0)
        mod = bb.get()

    BlockBuilder can also be used to construct neural networks with nn.Module API

    .. code-block:: python

        from tvm.relax.testing import nn

        n = tir.Var("n", "int64")
        input_size = 784
        hidden_sizes = [128, 32]
        output_size = 10
        bb = rx.BlockBuilder()

        with bb.function("main"):
            model = nn.Sequential(
                nn.Linear(input_size, hidden_sizes[0]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[1], output_size),
                nn.LogSoftmax(),
            )
            data = nn.Placeholder((n, input_size), name="data")
            output = model(data)
            params = [data] + model.parameters()
            builder.emit_func_output(output, params=params)
        mod = bb.get()
    """

    _current = None

    @staticmethod
    def current():
        """Returns the current BlockBuilder."""
        return BlockBuilder._current

    def __init__(self, mod: IRModule = None):
        self._blocks: List[BindingBlock] = []
        # a boolean flag that tracks if emit_func_output has been called
        self._is_emit_func_output_called = False
        self.__init_handle_by_constructor__(_ffi_api.BlockBuilderCreate, mod)  # type: ignore

    def _begin_dataflow_block(self) -> None:
        _ffi_api.BlockBuilderBeginDataflowBlock(self)  # type: ignore

    def _begin_binding_block(self) -> None:
        _ffi_api.BlockBuilderBeginBindingBlock(self)  # type: ignore

    def _end_block(self) -> BindingBlock:
        return _ffi_api.BlockBuilderEndBlock(self)  # type: ignore

    def _enter_function_scope(self, name, params, attrs):
        if BlockBuilder.current() is not None:
            raise RuntimeError("BlockBuilder does not allow nested functions.")
        BlockBuilder._current = self
        self._func_name = name
        self._func_params = params
        self._func_attrs = attrs
        self.begin_scope(params)
        self._begin_binding_block()

    def _exit_function_scope(self, exc_type, exc_val, exc_tb):
        # record
        is_emit_func_output_called = self._is_emit_func_output_called
        # recover to default state
        self._blocks = []
        self._is_emit_func_output_called = False
        BlockBuilder._current = None

        # NOTE: we must raise after we recover the state so future
        # block builder scoping functions correctly
        if exc_type is None:
            if not is_emit_func_output_called:
                raise RuntimeError("emit_func_output must be called in a relax function.")

    def _convert_te_arg(
        self, te_args: Any, tir_var_map: Dict[tir.Var, tir.PrimExpr]
    ) -> typing.Tuple[Any, List[tvm.te.Tensor]]:
        """Helper function used by `call_te` to convert Relax expressions to TE tensor.

        In the common case, the type of te_args is a Relax expression and is converted
        into a TE tensor.
        If te_args is a nested or recursive datatype (i.e list, dict, tvm.ir.Map, tvm.ir.Array),
        we recursive and convert any value of type Relax expression into a TE tensor.
        Common values of type int, float, and str are preserved.

        In dynamic shape cases, the passed in arguments may contain TIR variable.
        For example, the argument can be a Relax Var with TensorStructInfo, which
        has symbolic shape, or the argument can be a ShapeExpr with symbolic variables.
        To make the PrimFunc generated by `call_te` has independent variables with
        the caller Relax function, we will substitute the TIR variables in the input
        arguments with fresh ones, which is done by maintaining a TIR variable mapping.

        Parameters
        ----------
        te_args : Any
            Argument to convert to TE

        tir_var_map : Dict[tir.Var, tir.PrimExpr]
            The TIR variable mapping, which maps TIR variables on the Relax function
            side to the new set of variables used on the PrimFunc side.

        Returns
        -------
        ret : (Any, [tvm.te.Tensor])
            A tuple of the converted te_args, and a list of te tensors for each converted
            Relax expression
        """
        te_args_list = []

        def _copy_undefined_var(expr: tir.PrimExpr):
            def _visit_expr(e: tir.PrimExpr):
                if isinstance(e, tir.Var) and e not in tir_var_map:
                    new_var = tir.Var(e.name, e.dtype)
                    tir_var_map[e] = new_var

            tir.stmt_functor.post_order_visit(expr, _visit_expr)

        def _convert_te_arg_helper(arg):
            if isinstance(arg, Expr):  # type: ignore
                if isinstance(arg.struct_info, TensorStructInfo):
                    assert isinstance(
                        arg.struct_info.shape, ShapeExpr
                    ), "emit_te now only supports Tensor that has ShapeExpr shape"
                    for shape_value in arg.struct_info.shape.values:
                        _copy_undefined_var(shape_value)

                    arg = te_tensor(arg, tir_var_map)
                    te_args_list.append(arg)
                    return arg
                elif isinstance(arg.struct_info, ShapeStructInfo):
                    assert isinstance(
                        arg, ShapeExpr
                    ), "For Expr having ShapeStructInfo, emit_te now only supports ShapeExpr"
                    return [_convert_te_arg_helper(val) for val in arg.values]
                elif isinstance(arg.struct_info, PrimStructInfo):
                    return arg.value
            elif isinstance(arg, (list, tvm.ir.Array)):
                return [_convert_te_arg_helper(x) for x in arg]
            elif isinstance(arg, tuple):
                return tuple([_convert_te_arg_helper(x) for x in arg])
            elif isinstance(arg, (dict, tvm.ir.Map)):
                for key in arg:
                    assert isinstance(
                        key, str
                    ), "emit_te only supports dict with string as the key currently"
                return {k: _convert_te_arg_helper(arg[k]) for k in arg}
            elif isinstance(arg, tir.PrimExpr):
                _copy_undefined_var(arg)
                return tir.stmt_functor.substitute(arg, tir_var_map)
            elif isinstance(arg, (int, float, str, tvm.ir.Type, tvm.ir.Attrs)) or arg is None:
                return arg
            raise TypeError("not supported type in emit_te: {}".format(type(arg)))

        new_arg = _convert_te_arg_helper(te_args)
        return new_arg, te_args_list

    def _get_unbound_tir_vars(self, args: List[tvm.te.Tensor]) -> List[tvm.tir.Var]:
        """get unbound TIR vars (i.e TIR vars used in the shape but is not
        itself a dimension of a shape)"""
        bound_vars = set()
        used_vars = set()

        def _populate_used_vars(expr):
            if isinstance(expr, tvm.tir.Var):
                used_vars.add(expr)

        for x in args:
            for s in x.shape:
                tvm.tir.stmt_functor.post_order_visit(s, _populate_used_vars)
                if isinstance(s, tir.Var):
                    bound_vars.add(s)

        diff = used_vars - bound_vars
        return list(diff)

    def function(
        self,
        name: str,
        params: Optional[Union[Var, Tuple, List[Var]]] = None,
        attrs: Optional[Dict[str, Object]] = None,
    ) -> FunctionScope:
        """Annotate a Relax function.

        Parameters
        ----------
        name : str, optional
            The name of the function

        params : tvm.relax.Var | Tuple | List[tvm.relax.Var], optional
            The parameters of the function.
            If params is None, it means deferring initialization of function parameters
            until emit_func_output.

        attrs : Dict[str, Object], optional
            The function attrs

        Returns
        -------
        ret: FunctionScope
            A FunctionScope for building a Relax function node.
        """
        if not params:
            params = None
        elif isinstance(params, rx.Var):
            params = [params]
        elif isinstance(params, (list, tuple)):
            for param in params:
                if not isinstance(param, rx.Var):
                    raise TypeError(
                        "each element of function parameters must be of type tvm.relax.Var,\
                                    but got: {}".format(
                            type(param)
                        )
                    )
        if attrs is None:
            attrs = {}
        return FunctionScope(self, name, params, attrs)

    def testing_scope(self, def_vars: List[tir.Var]) -> TestingScope:
        """Start a scope for unit-testing purposes.

        Parameters
        ----------
        def_vars: List[tir.Var]
            List of symbolic variables that are marked as defined in scope.

        Returns
        -------
        ret: TestingScope
            A TestingScope to setup builder for emit and other purposes.
        """
        return TestingScope(self, def_vars)

    def dataflow(self) -> DataflowScope:
        """Annotate a Relax dataflow block.

        Returns
        -------
        ret: DataflowScope
            A DataflowScope for building a Relax dataflow block.
        """
        return DataflowScope(self)

    def emit(self, expr: Expr) -> Var:
        """Emit an expr.
        This infers the shape and type of the expr, create a variable,
        and bind the expr to the variable.

        Parameters
        ----------
        expr : tvm.relax.Expr
            The Expr to be emitted.

        Returns
        -------
        ret : tvm.relax.Var
            A newly created variable that gets bound to the input expr.
        """
        return _ffi_api.BlockBuilderEmit(self, expr)  # type: ignore

    def call_te(self, func: Callable, *args: Any, **kwargs: Any) -> Expr:
        """Generate a call node according to the te function.
        This function converts arguments from relax expression to te tensor,
        The callback func should return a te tensor or a list of te tensors.
        Please see detailed example in emit_te

        Parameters
        ----------
        func : Callable
            A function that returns a te tensor or a list of te tensors.

        args : Any, optional
            arguments passed to the function.

        kwargs : Any, optional
            The keyword arguments passed to the function.
            Note that the key "primfunc_name_hint" is reserved for passing name hint
            to the PrimFunc that gets generated.

        Returns
        -------
        ret : tvm.relax.Call
            A newly created call node
        """

        primfunc_name_hint = kwargs.pop("primfunc_name_hint", None)
        tir_var_map: Dict[tir.Var, tir.PrimExpr] = dict()
        new_args, te_arg_list = self._convert_te_arg(args, tir_var_map)
        new_kwargs, te_kwarg_list = self._convert_te_arg(kwargs, tir_var_map)

        te_args = te_arg_list + te_kwarg_list

        te_out = func(*new_args, **new_kwargs)
        assert isinstance(te_out, tvm.te.tensor.Tensor) or (
            isinstance(te_out, (tuple, list, tvm.ir.Array))
            and all(isinstance(t, tvm.te.tensor.Tensor) for t in te_out)
        ), "only support te.tensor or tuple/list/Array of te.tensor as function output"

        outs = [te_out] if isinstance(te_out, tvm.te.tensor.Tensor) else list(te_out)
        unbound_tir_vars = self._get_unbound_tir_vars(te_args + outs)

        inputs = [*te_args] + outs
        tir_func = tvm.te.create_relax_prim_func(inputs, unbound_tir_vars, "int64")

        tir_func = tir_func.without_attr("global_symbol")

        if primfunc_name_hint:
            gvar = self.add_func(tir_func, primfunc_name_hint)
        else:
            gvar = self.add_func(tir_func, func.__name__)

        call_args = [x.op.value for x in te_args]

        def _shape_with_old_tir_var(
            shape_values: List[tir.PrimExpr], tir_var_inverse_map: Dict[tir.Var, tir.PrimExpr]
        ):
            return ShapeExpr(
                [tir.stmt_functor.substitute(value, tir_var_inverse_map) for value in shape_values]
            )

        # Invert the TIR variable mapping, to convert the output shape back
        # with old set of variables.
        tir_var_inverse_map = {v: k for k, v in tir_var_map.items()}

        output_sinfo = [
            TensorStructInfo(_shape_with_old_tir_var(out.shape, tir_var_inverse_map), out.dtype)
            for out in outs
        ]

        # add arguments for extra parameters from unbound var
        if len(unbound_tir_vars) > 0:
            call = call_tir(
                gvar,
                call_args,
                output_sinfo,
                tir_vars=_shape_with_old_tir_var(unbound_tir_vars, tir_var_inverse_map),
            )
        else:
            call = call_tir(gvar, call_args, output_sinfo)
        return call

    def emit_te(self, func: Callable, *args: Any, **kwargs: Any) -> Var:
        """Emit a call node according to the te function.
        This function converts arguments from relax expression to te tensor,
        The callback func should return a te tensor or a list of te tensors.

        Parameters
        ----------
        func : Callable
            A function that returns a te tensor or a list of te tensors.

        args : Any, optional
            arguments passed to the function.

        kwargs : Any, optional
            The keyword arguments passed to the function.
            Note that the key "primfunc_name_hint" is reserved for passing name hint
            to the PrimFunc that gets generated.

        Returns
        -------
        ret : tvm.relax.Var
            A newly created variable that gets bound to the call code.

        Example
        -------

        .. code-block:: python

            bb = rx.BlockBuilder()
            n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
            x = rx.Var("x", rx.TensorStructInfo([n, m], "float32"))
            y = rx.Var("y", rx.TensorStructInfo([n, m], "float32"))

            def te_func(args, args_dict, msg):
                A = args[0]
                B = args_dict["B"]
                return te.compute((128, 128), lambda i, j: A[i, j] + B[i, j])

            with bb.function([x, y], "rx_func"):
                out = bb.emit_te(te_func, [x], {"B": y}, msg="hello")
                bb.emit_func_output(out)

        will result in TVMScript

        .. code-block:: python

            @tvm.script.ir_module
            class Module:
                @T.prim_func
                def te_func(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle,
                            var_compute: T.handle) -> None:
                    # function attr dict
                    T.func_attr({"tir.noalias": True})
                    m = T.var("int64")
                    n = T.var("int64")
                    rxplaceholder = T.match_buffer(var_rxplaceholder, [n, m], dtype="float32")
                    rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, [n, m], dtype="float32")
                    compute = T.match_buffer(var_compute, [128, 128], dtype="float32")
                    # body
                    # with T.block("root")
                    for i0, i1 in T.grid(128, 128):
                        with T.block("compute"):
                            i, j = T.axis.remap("SS", [i0, i1])
                            T.reads([rxplaceholder[i, j], rxplaceholder_1[i, j]])
                            T.writes([compute[i, j]])
                            compute[i, j] = rxplaceholder[i, j] + rxplaceholder_1[i, j]

                @R.function
                def rx_func(x: Tensor((n, m), "float32"), y: Tensor((n, m), "float32")) -> Tensor:
                    # block 0
                    gv = relax.call_tir("te_func", (x, y), R.Tensor((128, 128), "float32"))
                    return gv

        Example
        -------

        .. code-block:: python

            bb = relax.BlockBuilder()
            n = tir.Var("n", "int64")
            x = relax.Var("x", relax.TensorStructInfo([n], "float32"))
            y = relax.Var("y", relax.TensorStructInfo([n + 1], "float32"))

            def te_func(A):
                C = te.compute((n + 1), lambda i: A[i])
                return C

            with bb.function("rx_func", [x, y]):
                x1 = bb.emit_te(te_func, y)
                bb.emit_func_output(x1)

        will result in TVMScript

        .. code-block:: python

            @tvm.script.ir_module
            class Module:
                @T.prim_func
                def te_func(var_rxplaceholder: T.handle, var_compute: T.handle, n: T.int64) -> None:
                    rxplaceholder = T.match_buffer(var_rxplaceholder, [n + T.int64(1)],
                                                   dtype="float32")
                    compute = T.match_buffer(var_compute, [n + T.int64(1)], dtype="float32")
                    # body
                    # with T.block("root")
                    for i0 in T.serial(0, n + T.int64(1)):
                        with T.block("compute"):
                            i = T.axis.spatial(n + T.int64(1), i0)
                            T.reads([rxplaceholder[i]])
                            T.writes([compute[i]])
                            compute[i] = rxplaceholder[i]

                @R.function
                def rx_func(x: Tensor((n,), "float32"), y: Tensor(((n + 1),), "float32"))
                    -> Tensor(None, "float32", ndim=-1):
                    # block 0
                    gv = relax.call_tir(te_func, (y,), R.Tensor((n + 1,), "float32"), (n,))
                    return gv
        """
        return self.emit(self.call_te(func, *args, **kwargs))

    def match_cast(self, value: Expr, struct_info: StructInfo) -> Var:
        """Emit a MatchCast.

        Parameters
        ----------
        value : tvm.relax.Expr
            The value of the MatchCast to be emitted.

        struct_info : StructInfo
            The struct info to be matched.

        Returns
        -------
        ret : tvm.relax.Var
            A newly created variable that get bounds to be the casted result.
        """
        return _ffi_api.BlockBuilderEmitMatchCast(self, value, struct_info)  # type: ignore

    def emit_output(self, output: Union[Expr, Tuple, List[Expr]]) -> None:
        """Emit output for the current dataflow block or function.

        Parameters
        ----------
        output : Expr | Tuple | List[Expr]
            The output of the current block/function.

        Returns
        -------
        ret : tvm.relax.Var
            The return variable which gets bound to the output.
        """
        if isinstance(output, (list, tuple)):
            output = Tuple(output)
        return _ffi_api.BlockBuilderEmitOutput(self, output)  # type: ignore

    def emit_func_output(
        self,
        output: Union[Expr, Tuple, List[Expr]],
        params: Optional[Union[Var, Tuple, List[Var]]] = None,
    ) -> None:
        """Emit output for the function.

        Parameters
        ----------
        output : Expr | Tuple | List[Expr]
            The output of the current block/function.

        params : tvm.relax.Var | Tuple | List[tvm.relax.Var], optional
            The parameters of the function to be built.
            If params is None, it means the params have been initialized in the function with scope.

        Returns
        -------
        ret : tvm.relax.Var
            The return variable which gets bound to the output.
        """
        if self._is_emit_func_output_called:
            raise RuntimeError("emit_func_output must be called exactly once in a relax function.")
        self._is_emit_func_output_called = True

        if self._func_params is not None and params is not None:
            raise RuntimeError(
                "function parameters have been initialized in the function with scope."
            )

        if self._func_params is None and params is None:
            raise RuntimeError("Relax function must have parameter.")

        if self._func_params is None:
            self._func_params = params

        if BlockBuilder.current() is not self:
            raise RuntimeError("BlockBuilder._current must be self.")

        if isinstance(output, (list, tuple)):
            output = Tuple(output)

        block = self._end_block()
        if len(block.bindings) > 0:
            self._blocks.append(block)
        seqe = self.normalize(rx.SeqExpr(self._blocks, output))

        # do not specify ret_struct_info and let constructor deduce
        # from seqe.struct_info
        func = rx.Function(self._func_params, seqe)
        for key, value in self._func_attrs.items():
            func = func.with_attr(key, value)
        self.end_scope()
        self.add_func(func, self._func_name)

    def normalize(self, expr: Expr) -> Expr:
        """Normalize an Expr to complete its shape and type.

        Parameters
        ----------
        expr : Expr
            The input expr.

        Returns
        -------
        ret : Expr
            The expr with normalized shape and type.
        """
        return _ffi_api.BlockBuilderNormalize(self, expr)  # type: ignore

    def get(self) -> tvm.IRModule:
        """Return the IRModule being built.

        Returns
        -------
        ret : tvm.IRModule
            An IRModule with Relax and TIR functions being built.
        """
        return _ffi_api.BlockBuilderGetContextIRModule(self)  # type: ignore

    def get_unique_name(self, name_prefix: str) -> str:
        """Generate a unique name with a specified prefix.

        Parameters
        ----------
        name_hint : str
            The name prefix.

        Returns
        -------
        ret : str
            The generated name.
        """
        return _ffi_api.BlockBuilderGetUniqueName(self, name_prefix)  # type: ignore

    def add_func(self, func: BaseFunc, func_name: str) -> GlobalVar:
        """Add a Relax function or a TIR PrimFunc to the IRModule being built.

        Parameters
        ----------
        func : BaseFunc
            The function to be added.

        func_name : str
            The name of the function to be added.

        Returns
        -------
        gvar : GlobalVar
            The global var bound to the added function.
        """
        return _ffi_api.BlockBuilderAddFunction(self, func, func_name)  # type: ignore

    def update_func(self, gv: GlobalVar, updated_func: BaseFunc) -> None:
        """Add a Relax function or a TIR PrimFunc to the IRModule being built.

        Parameters
        ----------
        gv : GlobalVar
            The global var referring the function to be updated.

        updated_func : BaseFunc
            The updated function.
        """
        return _ffi_api.BlockBuilderUpdateFunction(self, gv, updated_func)  # type: ignore

    def current_block_is_dataflow(self) -> bool:
        """Check if the block being built is DataflowBlock or not.

        Returns
        -------
        ret : bool
            A boolean that indicates if the block being built is DataflowBlock or not.
        """
        return _ffi_api.BlockBuilderCurrentBlockIsDataFlow(self)  # type: ignore

    def emit_normalized(self, binding: Binding) -> None:
        """Emit an already normalized binding.

        Parameters
        ----------
        binding: Binding
            The binding to be emitted.
        """
        _ffi_api.BlockBuilderEmitNormalized(self, binding)  # type: ignore

    def lookup_binding(self, var: Var) -> Optional[Expr]:
        """Lookup a var in the binding table binding_table_.

        Parameters
        ----------
        var: Var
            The input var.

        Returns
        -------
        expr: Expr
            The Expr bound to the input var.
        """
        return _ffi_api.BlockBuilderLookupBinding(self, var)  # type: ignore

    def begin_scope(self, params: Optional[List[Var]] = None) -> None:
        """Begin a new scope, with optional parameters that
        are visible within the scope.

        Parameters
        ----------
        params: Optional[List[Var]]
            Parameters that are visible within the scope.

        Note
        ----
        This function should be called when new scope is introduced
        (function, seq) to properly track the variable availability
        and help the best effort deduction.
        """

        return _ffi_api.BlockBuilderBeginScope(self, params)  # type: ignore

    def end_scope(self) -> None:
        """End the current scope. Please see `begin_scope` for details"""

        return _ffi_api.BlockBuilderEndScope(self)  # type: ignore
