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
# pylint: disable=no-else-return, invalid-name, unused-argument, import-outside-toplevel
"""Developer API of constructing Relax AST."""

from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import tvm
from tvm import relax as rx
from tvm import tir
from tvm.ir.module import IRModule
from tvm.runtime import Object

from . import _ffi_api
from .expr import BaseFunc, Binding, BindingBlock, Expr, GlobalVar, Tuple, Var
from .op.base import call_tir, call_tir_with_grad
from .struct_info import StructInfo
from .utils import gen_call_tir_inputs


class FunctionScope(object):
    """Auxiliary scope for function"""

    def __init__(self, block_builder, name, params, attrs):
        self._bb = block_builder
        self._name = name
        self._params = params
        self._attrs = attrs

        # Blocks that have been collected within the function
        self._blocks = []
        # a boolean flag that tracks if emit_func_output has been called
        self._is_emit_func_output_called = False

    def __enter__(self):
        self._bb._enter_function_scope(self)

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
            self._bb._func._blocks.append(block)
        self._bb._begin_dataflow_block()

    def __exit__(self, ptype, value, trace):
        block = self._bb._end_block()
        if len(block.bindings) > 0:
            self._bb._func._blocks.append(block)
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

    _stack = []

    @staticmethod
    def current() -> Optional["BlockBuilder"]:
        """Returns the current BlockBuilder."""
        if BlockBuilder._stack:
            return BlockBuilder._stack[-1]
        else:
            return None

    def __init__(self, mod: IRModule = None):
        # Which functions are currently being defined
        self._func_stack: List[FunctionScope] = []
        self.__init_handle_by_constructor__(_ffi_api.BlockBuilderCreate, mod)  # type: ignore

    def _begin_dataflow_block(self) -> None:
        _ffi_api.BlockBuilderBeginDataflowBlock(self)  # type: ignore

    def _begin_binding_block(self) -> None:
        _ffi_api.BlockBuilderBeginBindingBlock(self)  # type: ignore

    def _end_block(self) -> BindingBlock:
        return _ffi_api.BlockBuilderEndBlock(self)  # type: ignore

    @property
    def _func(self):
        if self._func_stack:
            return self._func_stack[-1]
        else:
            raise RuntimeError(
                "Cannot access BlockBuilder._func when outside a bb._function() block"
            )

    def _enter_function_scope(self, func_scope):
        BlockBuilder._stack.append(self)
        self._func_stack.append(func_scope)
        self.begin_scope(func_scope._params)
        self._begin_binding_block()

    def _exit_function_scope(self, exc_type, exc_val, exc_tb):
        # record
        is_emit_func_output_called = self._func._is_emit_func_output_called
        # recover to default state
        self._func_stack.pop()

        assert BlockBuilder._stack
        assert BlockBuilder._stack[-1] is self
        BlockBuilder._stack.pop()

        # NOTE: we must raise after we recover the state so future
        # block builder scoping functions correctly
        if exc_type is None:
            if not is_emit_func_output_called:
                raise RuntimeError("emit_func_output must be called in a relax function.")

    def function(
        self,
        name: str,
        params: Optional[Union[Var, Tuple, List[Var]]] = None,
        attrs: Optional[Dict[str, Object]] = None,
        private: bool = False,
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

        private : bool, optional
            Whether the function is annotated as private.
            If the function is private, it will not have a global symbol attribute.
            If it is not private and not an inner function, then it will have
            a global symbol attribute (mapped to the function's name)

        Returns
        -------
        ret: FunctionScope
            A FunctionScope for building a Relax function node.
        """
        if isinstance(params, rx.Var):
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
        # The block builder does not permit nesting functions, per above comment,
        # so no further check should be needed
        if not private:
            attrs["global_symbol"] = name

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

    def _normalize_python_tuple(self, expr: Union[Expr, Sequence[Expr]]):
        """Internal utility function to convert to relax.Tuple

        The `emit`, `emit_output`, and `emit_func_output` can be
        called with python `list` or `tuple` objects.  These objects
        should be converted to `relax.Tuple` prior to calling an FFI
        function, as they would otherwise be converted to
        `tvm.runtime.Array`.  In addition, any nested tuple objects
        should be converted.
        """
        if isinstance(expr, (list, tuple)):
            return Tuple([self._normalize_python_tuple(element) for element in expr])
        else:
            return expr

    def emit(self, expr: Expr, name_hint: str = "") -> Var:
        """Emit an expr.
        This infers the shape and type of the expr, create a variable,
        and bind the expr to the variable.

        Parameters
        ----------
        expr : tvm.relax.Expr
            The Expr to be emitted.

        name_hint : str
            Name hint for the bound variable.

        Returns
        -------
        ret : tvm.relax.Var
            A newly created variable that gets bound to the input expr.
        """
        expr = self._normalize_python_tuple(expr)
        return _ffi_api.BlockBuilderEmit(self, expr, name_hint)  # type: ignore

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
            Note that the following keyword args are reserved:

                - 'primfunc_name_hint' for passing name hint to the PrimFunc
                  that gets generated.
                - 'primfunc_attrs' is reserved for passing func attributes to
                  be added to the PrimFunc that gets created.


        Returns
        -------
        ret : tvm.relax.Call
            A newly created call node
        """

        primfunc_name = kwargs.pop("primfunc_name_hint", None)
        tir_func, call_args, output_sinfo, tir_vars = gen_call_tir_inputs(func, *args, **kwargs)

        if not primfunc_name:
            primfunc_name = func.__name__
        gvar = self.add_func(tir_func, primfunc_name)

        return call_tir(gvar, call_args, output_sinfo, tir_vars)

    def call_te_with_grad(
        self,
        func: Callable,
        *args: Any,
        te_grad_name: str,
        te_grad_kwargs: Dict[str, Object] = None,
        **kwargs: Any,
    ) -> Expr:
        """Generate a call node according to the te function.
        This method will generate a call_tir_with_grad node, i.e. a call_tir node bound with a
        te gradient function (refered by te_grad_name).

        Parameters
        ----------
        func : Callable
            A function that returns a te tensor or a list of te tensors.

        args : Any, optional
            arguments passed to the function.

        te_grad_name : str
            The registered name of the te gradient function associated with the call_tir_with_grad
            node. Must be provided as a keyword argument.

        te_grad_kwargs : Dict[str, Object], optional
            The keyword arguments passed to the te gradient function.
            Optionally provided as a keyword argument. Default: {}.

        kwargs : Any, optional
            The keyword arguments passed to the function.
            Note that the following keyword args are reserved:

                - 'primfunc_name_hint' for passing name hint to the PrimFunc
                  that gets generated.
                - 'primfunc_attrs' is reserved for passing func attributes to
                  be added to the PrimFunc that gets created.

        Returns
        -------
        ret : tvm.relax.Call
            A newly created call node
        """

        primfunc_name = kwargs.pop("primfunc_name_hint", None)
        tir_func, call_args, output_sinfo, tir_vars = gen_call_tir_inputs(func, *args, **kwargs)

        if te_grad_kwargs is None:
            te_grad_kwargs = {}

        if not primfunc_name:
            primfunc_name = func.__name__
        gvar = self.add_func(tir_func, primfunc_name)

        return call_tir_with_grad(
            gvar, call_args, output_sinfo, te_grad_name, te_grad_kwargs, tir_vars
        )

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
                    m = T.int64()
                    n = T.int64()
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
        name_hint = kwargs.pop("name_hint", "")
        return self.emit(self.call_te(func, *args, **kwargs), name_hint=name_hint)

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

    def emit_output(self, output: Union[Expr, Tuple, List[Expr]], name_hint: str = "") -> Var:
        """Emit output for the current dataflow block or function.

        Parameters
        ----------
        output : Expr | Tuple | List[Expr]
            The output of the current block/function.

        name_hint : str
            Name hint for the bound variable.

        Returns
        -------
        ret : tvm.relax.Var
            The return variable which gets bound to the output.
        """
        output = self._normalize_python_tuple(output)
        return _ffi_api.BlockBuilderEmitOutput(self, output, name_hint)  # type: ignore

    def emit_func_output(
        self,
        output: Union[Expr, Tuple, List[Expr]],
        params: Optional[Union[Var, Tuple, List[Var]]] = None,
    ) -> GlobalVar:
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
        gvar: tvm.ir.GlobalVar

            A GlobalVar representing the function
        """
        if self._func._is_emit_func_output_called:
            raise RuntimeError("emit_func_output must be called exactly once in a relax function.")
        self._func._is_emit_func_output_called = True

        if self._func._params is not None and params is not None:
            raise RuntimeError(
                "function parameters have been initialized in the function with scope."
            )

        if self._func._params is None and params is None:
            raise RuntimeError("Relax function must have parameter.")

        if self._func._params is None:
            self._func._params = params

        if BlockBuilder.current() is not self:
            raise RuntimeError("BlockBuilder.current() must be self.")

        output = self._normalize_python_tuple(output)

        block = self._end_block()
        if len(block.bindings) > 0:
            self._func._blocks.append(block)

        seqe = rx.SeqExpr(self._func._blocks, output)

        # If the parameters were not provided as part of
        # `bb.function()`, then any variables provided from the params
        # are not in scope.  Otherwise, TIR variables used in dynamic
        # inputs are removed as undefined (e.g. Replacing
        # `R.Tensor(["batch_size"])` with `R.Tensor(ndims=1)`).
        self.begin_scope(self._func._params)
        try:
            seqe = self.normalize(seqe)
        finally:
            self.end_scope()

        # do not specify ret_struct_info and let constructor deduce
        # from seqe.struct_info
        func = rx.Function(self._func._params, seqe)
        for key, value in self._func._attrs.items():
            func = func.with_attr(key, value)
        self.end_scope()
        return self.add_func(func, self._func._name)

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
        """Return intermediate IRModule. For the situation where the IRModule is needed in the
        middle of a building process.

        Returns
        -------
        ret : tvm.IRModule
            An IRModule with Relax and TIR functions being built.
        """
        return _ffi_api.BlockBuilderGetContextIRModule(self)  # type: ignore

    def finalize(self) -> tvm.IRModule:
        """Finalize the building process and return the result IRModule.

        Possibly rename GlobalVars in the IRModule to ensure name uniqueness and the invariant:
        every public function has the same name as its "global_symbol" attribute.

        Note this method should be called only once at the end of the building process, since it may
        invalidate global vars previously returned by this builder.
        See also tvm.relax.transform.NormalizeGlobalVar.

        Returns
        -------
        ret : tvm.IRModule
            An IRModule with Relax and TIR functions being built.
        """
        return _ffi_api.BlockBuilderFinalize(self)  # type: ignore

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
        """Lookup a var in the binding table.

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
