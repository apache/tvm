"""Developer API of IR node builder make function."""
from __future__ import absolute_import as _abs

from . import api as _api
from . import stmt as _stmt
from . import expr as _expr
from . import make as _make
from . import ir_pass as _pass
from . import collections as _collections
from ._base import string_types
from ._ctypes._node import NodeGeneric

class WithScope(object):
    """Auxiliary scope  with"""
    def __init__(self, enter_value, exit_cb):
        self._enter_value = enter_value
        self._exit_cb = exit_cb

    def __enter__(self):
        return self._enter_value

    def __exit__(self, ptype, value, trace):
        self._exit_cb()


class BufferVar(NodeGeneric):
    """Buffer variable with content type, makes load store easily.

    Do not create it directly, create use IRBuilder.

    Examples
    --------
    In the follow example, x is BufferVar.
    :code:`x[0] = ...` directly emit a store to the IRBuilder,
    :code:`x[10]` translates to Load.

    .. code-block:: python

        # The following code generate IR for x[0] = x[
        ib = tvm.ir_builder.create()
        x = ib.pointer("float32")
        x[0] = x[10] + 1

    See Also
    --------
    IRBuilder.pointer
    IRBuilder.buffer_ptr
    IRBuilder.allocate
    """
    def __init__(self, builder, buffer_var, content_type):
        self._builder = builder
        self._buffer_var = buffer_var
        self._content_type = content_type

    def asnode(self):
        return self._buffer_var

    def __getitem__(self, index):
        return _make.Load(self._content_type, self._buffer_var, index)

    def __setitem__(self, index, value):
        value = _api.convert(value)
        if value.dtype != self._content_type:
            raise ValueError(
                "data type does not match content type %s vs %s" % (
                    value.dtype, self._content_type))
        self._builder.emit(_make.Store(self._buffer_var, value, index))


class IRBuilder(object):
    """Auxiliary builder to build IR for testing and dev.

    Examples
    --------
    .. code-block:: python

        ib = tvm.ir_builder.create()
        n = tvm.var("n")
        A = ib.allocate("float32", n, name="A")
        with ib.for_range(0, n, name="i") as i:
            with ib.if_scope((i % 2) == 0):
                A[i] = A[i] + 1
        # The result stmt.
        stmt = ib.get()
    """
    def __init__(self):
        self._seq_stack = [[]]

    def _pop_seq(self):
        """Pop sequence from stack"""
        seq = self._seq_stack.pop()
        if len(seq) == 0 or callable(seq[-1]):
            seq.append(_make.Evaluate(0))
        stmt = seq[-1]
        for s in reversed(seq[:-1]):
            if callable(s):
                stmt = s(stmt)
            else:
                assert isinstance(s, _stmt.Stmt)
                stmt = _make.Block(s, stmt)
        return stmt

    def emit(self, stmt):
        """Emit a statement to the end of current scope.

        Parameters
        ----------
        stmt : Stmt or callable.
           The statement to be emitted or callable that build stmt given body.
        """
        if isinstance(stmt, _expr.Call):
            stmt = _make.Evaluate(stmt)
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

            ib = tvm.ir_builder.create()
            i = tvm.var("i")
            x = ib.pointer("float32")
            ib.scope_attr(x, "storage_scope", "global")
            x[i] = x[i - 1] + 1
        """
        if isinstance(node, string_types):
            node = _make.StringImm(node)
        if isinstance(value, string_types):
            value = _make.StringImm(value)
        self.emit(lambda x: _make.AttrStmt(node, attr_key, value, x))

    def for_range(self, begin, end, name="i", dtype="int32"):
        """Create a for iteration scope.

        Parameters
        ----------
        begin : Expr
            The min iteration scope.

        end : Expr
            The end iteration scope

        name : str, optional
            The name of iteration variable

        dtype : str, optional
            The data type of iteration variable.

        Returns
        -------
        loop_scope : With.Scope of Var
            The for scope, when enters returns loop_var

        Examples
        --------
        .. code-block:: python

            ib = tvm.ir_builder.create()
            x = ib.pointer("float32")
            with ib.for_range(1, 10, name="i") as i:
                x[i] = x[i - 1] + 1
        """
        self._seq_stack.append([])
        loop_var = _api.var(name, dtype=dtype)
        extent = end if begin == 0 else _pass.Simplify(end - begin)
        def _exit_cb():
            self.emit(_make.For(
                loop_var, begin, extent, 0, 0, self._pop_seq()))
        return WithScope(loop_var, _exit_cb)

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

            ib = tvm.ir_builder.create()
            i = tvm.var("i")
            x = ib.pointer("float32")
            with ib.if_scope((i % 2) == 0):
                x[i] = x[i - 1] + 1
        """
        self._seq_stack.append([])
        def _exit_cb():
            self.emit(_make.IfThenElse(cond, self._pop_seq(), None))
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

            ib = tvm.ir_builder.create()
            i = tvm.var("i")
            x = ib.pointer("float32")
            with ib.if_scope((i % 2) == 0):
                x[i] = x[i - 1] + 1
            with ib.else_scope():
                x[i] = x[i - 1] + 2
        """
        if len(self._seq_stack[-1]) == 0:
            raise RuntimeError("else_scope can only follow an if_scope")
        prev = self._seq_stack[-1][-1]
        if not isinstance(prev, _stmt.IfThenElse) or prev.else_case:
            raise RuntimeError("else_scope can only follow an if_scope")
        self._seq_stack[-1].pop()
        self._seq_stack.append([])
        def _exit_cb():
            self.emit(_make.IfThenElse(prev.condition, prev.then_case, self._pop_seq()))
        return WithScope(None, _exit_cb)

    def allocate(self, dtype, shape, name="buf", scope=None):
        """Create a allocate statement.

        Parameters
        ----------
        dtype : str
            The content data type.

        shape : tuple of Expr
            The shape of array to be allocated.

        name : str, optional
            The name of the buffer.

        scope : str, optional
            The scope of the buffer.

        Returns
        -------
        buffer : BufferVar
            The buffer var representing the buffer.
        """
        buffer_var = _api.var(name, dtype="handle")
        if not isinstance(shape, (list, tuple, _collections.Array)):
            shape = [shape]
        if scope:
            self.scope_attr(buffer_var, "storage_scope", scope)
        self.emit(lambda x: _make.Allocate(
            buffer_var, dtype, shape, _api.const(1, dtype="uint1"), x))
        return BufferVar(self, buffer_var, dtype)

    def pointer(self, content_type, name="ptr"):
        """Create pointer variable with content type.

        Parameters
        ----------
        content_type : str
            The content data type.

        name : str, optional
            The name of the pointer.

        Returns
        -------
        ptr : BufferVar
            The buffer var representing the buffer.
        """
        buffer_var = _api.var(name, dtype="handle")
        return BufferVar(self, buffer_var, content_type)

    def buffer_ptr(self, buf):
        """Create pointer variable corresponds to buffer ptr.

        Parameters
        ----------
        buf : Buffer
            The buffer to be extracted.

        Returns
        -------
        ptr : BufferVar
            The buffer var representing the buffer.
        """
        return BufferVar(self, buf.data, buf.dtype)

    def get(self):
        """Return the builded IR.

        Returns
        -------
        stmt : Stmt
           The result statement.
        """
        seq = self._pop_seq()
        if len(self._seq_stack) != 0:
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
