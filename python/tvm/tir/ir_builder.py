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
"""Developer API of IR node builder make function."""
from tvm._ffi.base import string_types
from tvm.runtime import ObjectGeneric, DataType, convert, const
from tvm.ir import container as _container, PointerType, PrimType, Range

from . import buffer as _buffer
from . import stmt as _stmt
from . import expr as _expr
from . import op


class WithScope(object):
    """Auxiliary scope  with"""

    def __init__(self, enter_value, exit_cb):
        self._enter_value = enter_value
        self._exit_cb = exit_cb

    def __enter__(self):
        return self._enter_value

    def __exit__(self, ptype, value, trace):
        self._exit_cb()


class BufferVarBuilder(ObjectGeneric):
    """Helper to build Load/Store interactions with a buffer.

    The BufferVarBuilder gives array access into physical memory.
    Indices should be flat values, and are used in Load/Store nodes.

    Do not create a BufferVarBuilder directly.  Instead, use
    `IRBuilder.allocate` or `IRBuilder.pointer`.

    Examples
    --------
    In the follow example, x is BufferVarBuilder.
    :code:`x[0] = ...` directly emit a Store to the IRBuilder,
    :code:`x[10]` translates to Load.

    .. code-block:: python

        ib = tvm.tir.ir_builder.create()

        # One-dimensional buffer access
        x = ib.pointer("float32")
        x[0] = x[10] + 1

        # Implementing multi-dimensional array access using a linear index
        y = ib.allocate("float32", 32*32)
        y[(2*32) + 31] = 0.

    See Also
    --------
    IRBuilder.pointer
    IRBuilder.buffer_ptr
    IRBuilder.allocate
    IRBuilder.buffer_realize

    """

    def __init__(self, builder, buffer_var, content_type):
        self._builder = builder
        self._buffer_var = buffer_var
        self._content_type = content_type

    def asobject(self):
        return self._buffer_var

    @property
    def dtype(self):
        return self._content_type

    def _normalize_index(self, index):
        t = DataType(self._content_type)
        if t.lanes > 1:
            base = index * t.lanes
            stride = 1 if (not hasattr(base, "dtype")) else const(1, base.dtype)
            index = _expr.Ramp(base, stride, t.lanes)

        if isinstance(index, _expr.IterVar):
            index = index.var

        return index

    def __getitem__(self, index):
        index = self._normalize_index(index)
        return _expr.Load(self._content_type, self._buffer_var, index)

    def __setitem__(self, index, value):
        value = convert(value)
        if value.dtype != self._content_type:
            raise ValueError(
                "data type does not match content type %s vs %s" % (value.dtype, self._content_type)
            )

        index = self._normalize_index(index)
        self._builder.emit(_stmt.Store(self._buffer_var, value, index))


class BufferBuilder(ObjectGeneric):
    """Helper to build BufferLoad/BufferStore interactions with a buffer.

    The BufferBuilder gives multi-dimensional array access into
    logical memory.  Indices should have the same number of dimensions
    as the underlying buffer.  Read/writes to the BufferBuilder
    correspond to BufferLoad/BufferStore nodes.  For physical memory
    access, see BufferVarBuilder.

    Do not create a BufferBuilder directly.  Instead, use
    `IRBuilder.buffer_realize` or `IRBuilder.buffer_ptr`.

    Examples
    --------
    In the follow example, x is BufferVarBuilder.
    :code:`x[0] = ...` directly emit a BufferStore to the IRBuilder,
    :code:`x[10]` translates to BufferLoad.

    .. code-block:: python

        ib = tvm.tir.ir_builder.create()
        # One-dimensional buffer access
        x = ib.buffer_realize("float32", 16)
        x[0] = x[10] + 1.0

        # Multi-dimensional buffer access
        y = ib.buffer_realize("float32", (16, 32))
        # Array access using a multidimensional index
        y[2, 31] = 0.0

    See Also
    --------
    IRBuilder.pointer
    IRBuilder.buffer_ptr
    IRBuilder.allocate
    IRBuilder.buffer_realize

    """

    def __init__(self, builder, buffer, content_type):
        self._builder = builder
        self._buffer = buffer
        self._content_type = content_type

    def asobject(self):
        return self._buffer

    @property
    def dtype(self):
        return self._content_type

    def _normalize_index(self, index):
        try:
            index = [*index]
        except TypeError:
            index = [index]

        t = DataType(self._content_type)
        if t.lanes > 1:
            base = index[-1] * t.lanes
            stride = 1 if (not hasattr(base, "dtype")) else const(1, base.dtype)
            index[-1] = _expr.Ramp(base, stride, t.lanes)

        index = [x.var if isinstance(x, _expr.IterVar) else x for x in index]

        return index

    def __getitem__(self, index):
        index = self._normalize_index(index)
        return _expr.BufferLoad(self._buffer, index)

    def __setitem__(self, index, value):
        index = self._normalize_index(index)

        value = convert(value)
        if value.dtype != self._content_type:
            raise ValueError(
                "data type does not match content type %s vs %s" % (value.dtype, self._content_type)
            )

        self._builder.emit(_stmt.BufferStore(self._buffer, value, index))


class IRBuilder(object):
    """Auxiliary builder to build IR for testing and dev.

    Examples
    --------
    .. code-block:: python

        ib = tvm.tir.ir_builder.create()
        n = te.var("n")
        A = ib.allocate("float32", n, name="A")
        with ib.for_range(0, n, name="i") as i:
            with ib.if_scope((i % 2) == 0):
                A[i] = A[i] + 1
        # The result stmt.
        stmt = ib.get()
    """

    def __init__(self):
        self._seq_stack = [[]]  # type: ignore
        self.nidx = 0

    def _pop_seq(self):
        """Pop sequence from stack"""
        seq = self._seq_stack.pop()
        if not seq or callable(seq[-1]):
            seq.append(_stmt.Evaluate(0))
        seqwrap = lambda x: x[0] if len(x) == 1 else _stmt.SeqStmt(list(reversed(x)))
        ret_seq = [seq[-1]]

        for s in reversed(seq[:-1]):
            if callable(s):
                ret_seq = [s(seqwrap(ret_seq))]
            else:
                assert isinstance(s, _stmt.Stmt)
                ret_seq.append(s)
        return seqwrap(ret_seq)

    def emit(self, stmt):
        """Emit a statement to the end of current scope.

        Parameters
        ----------
        stmt : Stmt or callable.
           The statement to be emitted or callable that build stmt given body.
        """
        if isinstance(stmt, _expr.Call):
            stmt = _stmt.Evaluate(stmt)
        assert isinstance(stmt, _stmt.Stmt) or callable(stmt)
        self._seq_stack[-1].append(stmt)

    def scope_attr(self, node, attr_key, value):
        """Create an AttrStmt at current scope.

        Parameters
        ----------
        attr_key : str
            The key of the attribute type.

        node : Node
            The attribute node to annottate on.

        value : Expr
            Attribute value.

        Examples
        --------
        .. code-block:: python

            ib = tvm.tir.ir_builder.create()
            i = te.var("i")
            x = ib.pointer("float32")
            ib.scope_attr(x, "storage_scope", "global")
            x[i] = x[i - 1] + 1
        """
        if isinstance(node, string_types):
            node = _expr.StringImm(node)
        if isinstance(value, string_types):
            value = _expr.StringImm(value)
        # thread_extent could be zero for dynamic workloads
        if attr_key == "thread_extent":
            value = op.max(1, value)
        self.emit(lambda x: _stmt.AttrStmt(node, attr_key, value, x))

    def for_range(self, begin, end, name="i", dtype="int32", kind="serial"):
        """Create a for iteration scope.

        Parameters
        ----------
        begin : Expr
            The min iteration scope.

        end : Expr
            The end iteration scope

        name : str, optional
            The name of iteration variable, if no input names,
            using typical index names i, j, k, then i_nidx

        dtype : str, optional
            The data type of iteration variable.

        kind : str, optional
            The special tag on the for loop.

        Returns
        -------
        loop_scope : With.Scope of Var
            The for scope, when enters returns loop_var

        Examples
        --------
        .. code-block:: python

            ib = tvm.tir.ir_builder.create()
            x = ib.pointer("float32")
            with ib.for_range(1, 10, name="i") as i:
                x[i] = x[i - 1] + 1
        """
        if name == "i":
            name = chr(ord(name) + self.nidx) if self.nidx < 3 else name + "_" + str(self.nidx - 3)
            self.nidx += 1
        self._seq_stack.append([])
        loop_var = _expr.Var(name, dtype=dtype)
        extent = end if begin == 0 else (end - begin)

        def _exit_cb():
            if kind == "serial":
                kind_id = _stmt.ForKind.SERIAL
            elif kind == "parallel":
                kind_id = _stmt.ForKind.PARALLEL
            elif kind == "vectorize":
                kind_id = _stmt.ForKind.VECTORIZED
            elif kind == "unroll":
                kind_id = _stmt.ForKind.UNROLLED
            else:
                raise ValueError("Unknown kind")
            self.emit(_stmt.For(loop_var, begin, extent, kind_id, self._pop_seq()))

        return WithScope(loop_var, _exit_cb)

    def while_loop(self, condition):
        """Create a while loop scope.

        Parameters
        ----------
        condition : Expr
            The termination condition.

        Returns
        -------
        loop_scope : With.Scope of Var
            The while scope.

        Examples
        --------
        .. code-block:: python

            ib = tvm.tir.ir_builder.create()
            iterations = ib.allocate("int32", 1, name="iterations", scope="local")
            with ib.while_loop(iterations[0] < 10):
                iterations[0] += 1
        """
        self._seq_stack.append([])

        def _exit_cb():
            self.emit(_stmt.While(condition, self._pop_seq()))

        return WithScope(None, _exit_cb)

    def if_scope(self, cond):
        """Create an if scope.

        Parameters
        ----------
        cond : Expr
            The condition.

        Returns
        -------
        if_scope : WithScope
           The result if scope.

        Examples
        --------
        .. code-block:: python

            ib = tvm.tir.ir_builder.create()
            i = te.var("i")
            x = ib.pointer("float32")
            with ib.if_scope((i % 2) == 0):
                x[i] = x[i - 1] + 1
        """
        self._seq_stack.append([])

        def _exit_cb():
            self.emit(_stmt.IfThenElse(cond, self._pop_seq(), None))

        return WithScope(None, _exit_cb)

    def else_scope(self):
        """Create an else scope.

        This can only be used right after an if scope.

        Returns
        -------
        else_scope : WithScope
           The result else scope.

        Examples
        --------
        .. code-block:: python

            ib = tvm.tir.ir_builder.create()
            i = te.var("i")
            x = ib.pointer("float32")
            with ib.if_scope((i % 2) == 0):
                x[i] = x[i - 1] + 1
            with ib.else_scope():
                x[i] = x[i - 1] + 2
        """
        if not self._seq_stack[-1]:
            raise RuntimeError("else_scope can only follow an if_scope")
        prev = self._seq_stack[-1][-1]
        if not isinstance(prev, _stmt.IfThenElse) or prev.else_case:
            raise RuntimeError("else_scope can only follow an if_scope")
        self._seq_stack[-1].pop()
        self._seq_stack.append([])

        def _exit_cb():
            self.emit(_stmt.IfThenElse(prev.condition, prev.then_case, self._pop_seq()))

        return WithScope(None, _exit_cb)

    def new_scope(self):
        """Create new scope,

        this is useful to set boundary of attr and allocate.

        Returns
        -------
        new_scope : WithScope
           The result new scope.
        """
        self._seq_stack.append([])

        def _exit_cb():
            self.emit(self._pop_seq())

        return WithScope(None, _exit_cb)

    def let(self, var_name, value):
        """Create a new let stmt binding.

        Parameters
        ----------
        var_name : str
            The name of the variable

        value : PrimExpr
            The value to be bound

        Returns
        -------
        var : tvm.tir.Var
           The var that can be in for future emits.
        """
        var = _expr.Var(var_name, dtype=value.dtype)
        self.emit(lambda x: _stmt.LetStmt(var, value, x))
        return var

    def allocate(self, dtype, extent, name="buf", scope=""):
        """Create a allocate statement.

        Parameters
        ----------
        dtype : str
            The content data type.

        extent : Expr
            The size of array to be allocated.

        name : str, optional
            The name of the buffer.

        scope : str, optional
            The scope of the buffer.

        Returns
        -------
        buffer : BufferVar
            The buffer var representing the buffer.
        """
        buffer_var = _expr.Var(name, PointerType(PrimType(dtype), scope))
        self.emit(lambda x: _stmt.Allocate(buffer_var, dtype, extent, const(1, dtype="uint1"), x))
        return BufferVarBuilder(self, buffer_var, dtype)

    def buffer_realize(self, dtype, shape, name="buf", scope=""):
        """Create a BufferRealize statement.

        Parameters
        ----------
        dtype : str
            The content data type.

        shape : Union[Expr, List[Expr], Tuple[Expr]]
            The shape of array to be allocated.

        name : str, optional
            The name of the buffer.

        scope : str, optional
            The scope of the buffer.

        Returns
        -------
        buffer : BufferBuilder
            The buffer var representing the buffer.
        """
        buffer = _buffer.decl_buffer(shape, dtype=dtype, name=name, scope=scope)
        if not isinstance(shape, (list, tuple, _container.Array)):
            shape = [shape]

        bounds = [Range(0, dim_extent) for dim_extent in shape]

        self.emit(lambda x: _stmt.BufferRealize(buffer, bounds, True, x))
        return BufferBuilder(self, buffer, dtype)

    def pointer(self, content_type, name="ptr", scope=""):
        """Create pointer variable with content type.

        Parameters
        ----------
        content_type : str
            The content data type.

        name : str, optional
            The name of the pointer.

        scope : str, optional
            The scope of the pointer.

        Returns
        -------
        ptr : BufferVarBuilder
            The buffer var representing the buffer.
        """
        buffer_var = _expr.Var(name, PointerType(PrimType(content_type), scope))
        return BufferVarBuilder(self, buffer_var, content_type)

    def buffer_ptr(self, buf):
        """Create a handle to interact with the buffer specified.

        Parameters
        ----------
        buf : Buffer
            The buffer to be extracted.

        shape : Tuple
            Optional shape of the buffer. Overrides existing buffer shape.

        Returns
        -------
        ptr : BufferBuilder
            The buffer var representing the buffer.
        """
        return BufferBuilder(self, buf, buf.dtype)

    def likely(self, expr):
        """Add likely tag for expression.
        Parameters
        ----------
        expr : Expr
            The expression. Usually a condition expression.
        Returns
        -------
        expr : Expr
            The expression will likely tag.
        """
        return _expr.Call(expr.dtype, "tir.likely", [expr])

    def get(self):
        """Return the builded IR.

        Returns
        -------
        stmt : Stmt
           The result statement.
        """
        seq = self._pop_seq()
        if self._seq_stack:
            raise RuntimeError("cannot call get inside construction scope")
        return seq


def create():
    """Create a new IRBuilder

    Returns
    -------
    builder : IRBuilder
        The created IRBuilder
    """
    return IRBuilder()
