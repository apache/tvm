"""Functions defined in TVM."""
# pylint: disable=invalid-name,unused-import,redefined-builtin
from __future__ import absolute_import as _abs

from numbers import Integral as _Integral

from ._ffi.base import string_types
from ._ffi.node import register_node, NodeBase
from ._ffi.node import convert_to_node as _convert_to_node
from ._ffi.function import Function
from ._ffi.function import _init_api, register_func, get_global_func, extract_ext_funcs
from ._ffi.function import convert_to_tvm_func as _convert_tvm_func
from ._ffi.runtime_ctypes import TVMType
from . import _api_internal
from . import make as _make
from . import expr as _expr
from . import tensor as _tensor
from . import schedule as _schedule
from . import container as _container
from . import tag as _tag

int8 = "int8"
int32 = "int32"
float32 = "float32"
handle = "handle"


def min_value(dtype):
    """minimum value of dtype"""
    return _api_internal._min_value(dtype)


def max_value(dtype):
    """maximum value of dtype"""
    return _api_internal._max_value(dtype)


def const(value, dtype=None):
    """construct a constant"""
    if dtype is None:
        if isinstance(value, _Integral):
            dtype = 'int32'
        else:
            dtype = 'float32'
    return _api_internal._const(value, dtype)


def convert(value):
    """Convert value to TVM node or function.

    Parameters
    ----------
    value : python value

    Returns
    -------
    tvm_val : Node or Function
        Converted value in TVM
    """
    if isinstance(value, (Function, NodeBase)):
        return value

    if callable(value):
        return _convert_tvm_func(value)

    return _convert_to_node(value)


def load_json(json_str):
    """Load tvm object from json_str.

    Parameters
    ----------
    json_str : str
        The json string

    Returns
    -------
    node : Node
        The loaded tvm node.
    """
    return _api_internal._load_json(json_str)


def save_json(node):
    """Load tvm object as json string.

    Parameters
    ----------
    node : Node
        A TVM Node object to be saved.

    Returns
    -------
    json_str : str
        Saved json string.
    """
    return _api_internal._save_json(node)


def var(name="tindex", dtype=int32):
    """Create a new variable with specified name and dtype

    Parameters
    ----------
    name : str
        The name

    dtype : int
        The data type

    Returns
    -------
    var : Var
        The result symbolic variable.
    """
    return _api_internal._Var(name, dtype)


def any(*args):
    """Create a new experssion of the union of all conditions in the arguments

    Parameters
    ----------
    args : list
        List of symbolic boolean expressions

    Returns
    -------
    expr: Expr
        Expression
    """
    if not args:
        raise ValueError("Any must take at least 1 argument")
    if len(args) == 1:
        return args[0]
    ret = _make.Or(args[0], args[1])
    for i in range(2, len(args)):
        ret = _make.Or(ret, args[i])
    return ret


def all(*args):
    """Create a new experssion of the intersection of all conditions in the
      arguments

    Parameters
    ----------
    args : list
        List of symbolic boolean expressions

    Returns
    -------
    expr: Expr
        Expression
    """
    if not args:
        raise ValueError("Any must take at least 1 argument")
    if len(args) == 1:
        return args[0]
    ret = _make.And(args[0], args[1])
    for i in range(2, len(args)):
        ret = _make.And(ret, args[i])
    return ret


def placeholder(shape, dtype=None, name="placeholder"):
    """Construct an empty tensor object.

    Parameters
    ----------
    shape: Tuple of Expr
        The shape of the tensor

    dtype: str, optional
        The data type of the tensor

    name: str, optional
        The name hint of the tensor

    Returns
    -------
    tensor: Tensor
        The created tensor
    """
    shape = (shape,) if isinstance(shape, _expr.Expr) else shape
    dtype = float32 if dtype is None else dtype
    return _api_internal._Placeholder(
        shape, dtype, name)


def compute(shape, fcompute, name="compute", tag=""):
    """Construct a new tensor by computing over the shape domain.

    The compute rule is result[axis] = fcompute(axis)

    Parameters
    ----------
    shape: Tuple of Expr
        The shape of the tensor

    fcompute: lambda function of indices-> value
        Specifies the input source expression

    name: str, optional
        The name hint of the tensor

    Returns
    -------
    tensor: Tensor
        The created tensor
    """
    if _tag.TagScope.current is not None:
        if tag != "":
            raise ValueError("nested tag is not allowed for now")
        tag = _tag.TagScope.current.tag
    shape = (shape,) if isinstance(shape, _expr.Expr) else shape
    ndim = len(shape)
    code = fcompute.__code__

    if fcompute.__code__.co_argcount == 0:
        arg_names = ["i%d" % i for i in range(ndim)]
    else:
        arg_names = code.co_varnames[:code.co_argcount]

    if ndim != len(arg_names):
        raise ValueError("fcompute do not match dimension, ndim=%d" % ndim)

    dim_var = [_IterVar((0, s), x, 0) for x, s in zip(arg_names, shape)]
    body = fcompute(*[v.var for v in dim_var])
    if not isinstance(body, (list, tuple)):
        body = [body]
    body = convert(body)
    op_node = _api_internal._ComputeOp(
        name, tag, dim_var, body)
    num = op_node.num_outputs
    outputs = tuple(op_node.output(i) for i in range(num))
    return outputs[0] if num == 1 else outputs


def scan(init, update, state_placeholder, inputs=None, name="scan", tag=""):
    """Construct new tensors by scanning over axis.

    Parameters
    ----------
    init: Tensor or list of Tensor
        The initial condition of first init.shape[0] timestamps

    update: Tensor or list of Tensor
        The update rule of the scan given by symbolic tensor.

    state_placeholder: Tensor or list of Tensor
        The placeholder variables used by update.

    inputs: Tensor or list of Tensor, optional
        The list of inputs to the scan. This is not required, but can
        be useful for the compiler to detect scan body faster.

    name: str, optional
        The name hint of the tensor

    Returns
    -------
    tensor: Tensor or list of Tensors
        The created tensor or tuple of tensors it it contains multiple outputs.

    Example
    -------
    .. code-block:: python

      # The following code is equivalent to numpy.cumsum
      m = tvm.var("m")
      n = tvm.var("n")
      X = tvm.placeholder((m, n), name="X")
      s_state = tvm.placeholder((m, n))
      s_init = tvm.compute((1, n), lambda _, i: X[0, i])
      s_update = tvm.compute((m, n), lambda t, i: s_state[t-1, i] + X[t, i])
      res = tvm.scan(s_init, s_update, s_state, X)
    """
    if _tag.TagScope.current is not None:
        if tag != "":
            raise ValueError("nested tag is not allowed for now")
        tag = _tag.TagScope.current.tag
    if isinstance(init, _tensor.Tensor):
        init = [init]
    if isinstance(update, _tensor.Tensor):
        update = [update]
    if isinstance(state_placeholder, _tensor.Tensor):
        state_placeholder = [state_placeholder]
    if isinstance(inputs, _tensor.Tensor):
        inputs = [inputs]
    if inputs is None:
        inputs = []
    if len(init) != len(update) or len(init) != len(state_placeholder):
        raise ValueError("init, update, state_placeholder must have same length")
    axis = _IterVar((init[0].shape[0], update[0].shape[0]), "%s.idx" % name, 3)
    op = _api_internal._ScanOp(name, tag, axis, init, update,
                               state_placeholder, inputs)
    res = [op.output(i) for i in range(len(update))]
    return res[0] if len(res) == 1 else res


def extern(shape, inputs, fcompute, name="extern", dtype=None, tag=""):
    """Compute several tensor via extern function.

    Parameters
    ----------
    shape: tuple or list of tuples.
        The shape of the outputs.

    inputs: list of Tensor
        The inputs

    fcompute: lambda function of inputs, outputs-> stmt
        Specifies the IR statement to do the computation.
        See the following note for function signature of fcompute

        .. note::
             **Parameters**

             - **ins** (list of :any:`Buffer`) - Placeholder for each inputs
             - **outs** (list of :any:`Buffer`) - Placeholder for each outputs

             **Returns**

             - **stmt** (:any:`Stmt`) - The statement that carries out array computation.

    name: str, optional
        The name hint of the tensor

    dtype: str or list of str, optional
        The data types of outputs,
        by default dtype will be same as inputs.

    Returns
    -------
    tensor: Tensor or list of Tensors
        The created tensor or tuple of tensors it it contains multiple outputs.

    Example
    -------
    In the code below, C is generated by calling external PackedFunc
    `tvm.contrib.cblas.matmul`

    .. code-block:: python

        A = tvm.placeholder((n, l), name='A')
        B = tvm.placeholder((l, m), name='B')
        C = tvm.extern((n, m), [A, B],
                       lambda ins, outs: tvm.call_packed(
                          "tvm.contrib.cblas.matmul",
                            ins[0], ins[1], outs[0], 0, 0), name="C")
    """
    if _tag.TagScope.current is not None:
        if tag != "":
            raise ValueError("nested tag is not allowed for now")
        tag = _tag.TagScope.current.tag
    shape = (shape,) if isinstance(shape, (_expr.Expr, _Integral)) else shape
    shape = [shape] if isinstance(shape[0], (_expr.Expr, _Integral)) else shape
    input_placeholders = []
    output_placeholders = []
    types = set()
    for t in inputs:
        if not isinstance(t, _tensor.Tensor):
            raise ValueError("expect inputs to be tensor")
        input_placeholders.append(
            decl_buffer(t.shape, t.dtype, t.op.name))
        types.add(t.dtype)

    if dtype is None:
        if len(types) != 1:
            raise ValueError("Cannot infer output type, please provide dtype argument")
        infered_type = types.pop()
        dtype = [infered_type for _ in shape]
    if isinstance(dtype, str):
        dtype = [dtype]

    for shp, dt in zip(shape, dtype):
        output_placeholders.append(decl_buffer(shp, dt, name))
    body = fcompute(input_placeholders, output_placeholders)
    if isinstance(body, _expr.Expr):
        body = _make.Evaluate(body)

    op = _api_internal._ExternOp(name, tag, inputs, input_placeholders,
                                 output_placeholders, body)
    res = [op.output(i) for i in range(len(output_placeholders))]
    return res[0] if len(res) == 1 else res


def decl_buffer(shape,
                dtype=None,
                name="buffer",
                data=None,
                strides=None,
                elem_offset=None,
                scope="",
                data_alignment=-1,
                offset_factor=0):
    """Decleare a new symbolic buffer.

    Normally buffer is created automatically during lower and build.
    This is only needed if user want to specify their own buffer layout.

    See the note below for detailed discussion on usage of buffer.

    Parameters
    ----------
    shape : tuple of Expr
        The shape of the buffer.

    dtype : str, optional
        The data type of the buffer.

    name : str, optional
        The name of the buffer.

    data : Var, optional
        The data pointer in the buffer.

    strides: array of Expr
        The stride of the buffer.

    elem_offset: Expr, optional
        The beginning offset of the array to data.
        In terms of number of elements of dtype.

    scope: str, optional
        The storage scope of the buffer, if not global.
        If scope equals empty string, it means it is global memory.

    data_alignment: int, optional
        The alignment of data pointer in bytes.
        If -1 is passed, the alignment will be set to TVM's internal default.

    offset_factor: int, optional
        The factor of elem_offset field, when set,
        elem_offset is required to be multiple of offset_factor.
        If 0 is pssed, the alignment will be set to 1.
        if non-zero is passed, we will created a Var for elem_offset if elem_offset is not None.

    Returns
    -------
    buffer : Buffer
        The created buffer

    Note
    ----
    Buffer data structure reflects the DLTensor structure in dlpack.
    While DLTensor data structure is very general, it is usually helpful
    to create function that only handles specific case of data structure
    and make compiled function benefit from it.

    If user pass strides and elem_offset is passed as None
    when constructing the function, then the function will be specialized
    for the DLTensor that is compact and aligned.
    If user pass a fully generic symbolic array to the strides,
    then the resulting function becomes fully generic.
    """
    shape = (shape,) if isinstance(shape, (_expr.Expr, _Integral)) else shape
    dtype = float32 if dtype is None else dtype
    strides = () if strides is None else strides
    if offset_factor != 0 and elem_offset is None:
        elem_offset = var('%s_elem_offset' % name, shape[0].dtype)
    if data is None:
        data = var(name, "handle")
    return _api_internal._Buffer(
        data, dtype, shape, strides, elem_offset, name, scope,
        data_alignment, offset_factor)


def _IterVar(dom, name, iter_type, thread_tag=''):
    """Internal function to create IterVar

    Parameters
    ----------
    dom : Range
        The domain of iteration.

    name : str
        The name of iteration variable.

    iter_type : int
        The type of iteration.

    thread_tag : str
        The thread tag of the iteration variable.

    Returns
    -------
    iter_var : IterVar
       The result itervar
    """
    if dom is not None:
        if isinstance(dom, (list, tuple)):
            if len(dom) != 2:
                raise TypeError("need to be list of ranges")
            dom = Range(dom[0], dom[1])

        if not isinstance(dom, _container.Range):
            raise TypeError("dom need to be Range")
    name = name if name else 'iter'
    v = var(name)
    return _api_internal._IterVar(dom, v, iter_type, thread_tag)


def thread_axis(dom=None, tag='', name=''):
    """Create a new IterVar to represent thread index.

    Parameters
    ----------
    dom : Range or str
        The domain of iteration
        When str is passed, dom is set to None and str is used as tag

    tag : str, optional
        The thread tag

    name : str, optional
        The name of the var.

    Returns
    -------
    axis : IterVar
        The thread itervar.
    """
    if isinstance(dom, string_types):
        tag, dom = dom, None
    if not tag:
        raise ValueError("tag must be given as Positional or keyword argument")
    name = name if name else tag
    return _IterVar(dom, name, 1, tag)


def reduce_axis(dom, name="rv"):
    """Create a new IterVar for reduction.

    Parameters
    ----------
    dom : Range
        The domain of iteration.

    name : str
        The name of the variable.

    Returns
    -------
    axis : IterVar
        An iteration variable representing the value.
    """
    return _IterVar(dom, name, 2)


def select(cond, t, f):
    """Construct a select branch
    Parameters
    ----------
    cond : Expr
        The condition
    t : Expr
        The result expression if cond is true.
    f : Expr
        The result expression if cond is false.

    Returns
    -------
    node : Node
        The tvm.expr.Select node
    """
    return _make.Select(convert(cond), convert(t), convert(f))

def comm_reducer(fcombine, fidentity, name="reduce"):
    """Create a commutative reducer for reduction.

    Parameters
    ----------
    fcombine : function(Expr -> Expr -> Expr)
        A binary function which takes two Expr as input to return a Expr.

    fidentity : function(str -> Expr)
        A function which takes a type string as input to return a const Expr.

    Returns
    -------
    reducer : function
        A function which creates a reduce expression over axis.
        There are two ways to use it:

        1. accept (expr, axis, where) to produce an Reduce Expr on
           specified axis;
        2. simply use it with multiple Exprs.

    Example
    -------
    .. code-block:: python

        n = tvm.var('n')
        m = tvm.var('m')
        mysum = tvm.comm_reducer(lambda x, y: x+y,
            lambda t: tvm.const(0, dtype=t), name="mysum")
        A = tvm.placeholder((n, m), name='A')
        k = tvm.reduce_axis((0, m), name='k')
        B = tvm.compute((n,), lambda i: mysum(A[i, k], axis=k), name='B')
    """
    def _reduce_directly(*args):
        num = len(args)
        # process `where` is None
        if num == 3 and args[2] is None:
            num = 2
        res = args[0]
        for i in range(num-1):
            res = fcombine(res, args[i+1])
        return res

    def _make_reduce(expr, axis, where=None):
        code = fcombine.__code__
        assert fcombine.__code__.co_argcount == 2
        expr = convert(expr)
        if isinstance(expr, _container.Array):
            size = len(expr)
            larr = []
            rarr = []
            dtypes = []
            for i in range(size):
                dtype = expr[i].dtype
                dtypes.append(dtype)
                lname = code.co_varnames[0] + '_' + str(i)
                larr.append(var(lname, dtype))
                rname = code.co_varnames[1] + '_' + str(i)
                rarr.append(var(rname, dtype))
            lhs = convert(larr)
            rhs = convert(rarr)
            result = fcombine(lhs, rhs)
            id_elem = fidentity(*dtypes)
        else:
            assert isinstance(expr, _expr.Expr)
            size = 1
            dtype = expr.dtype
            lvar = var(code.co_varnames[0], dtype)
            rvar = var(code.co_varnames[1], dtype)
            result = [fcombine(lvar, rvar)]
            id_elem = [fidentity(dtype)]
            lhs = convert([lvar])
            rhs = convert([rvar])
            expr = convert([expr])
        result = convert(result)
        id_elem = convert(id_elem)
        combiner = _make.CommReducer(lhs, rhs, result, id_elem)
        axis = convert(axis if isinstance(axis, (list, tuple)) else [axis])
        if where is None:
            where = convert(True)
        outputs = tuple(_make.Reduce(combiner, expr, axis, where, i)
                        for i in range(size))
        return outputs[0] if size == 1 else outputs

    # pylint: disable=keyword-arg-before-vararg
    def reducer(expr, axis, where=None, *args):
        if isinstance(axis, (_schedule.IterVar, list, tuple)):
            assert not args
            return _make_reduce(expr, axis, where)
        if where is None:
            assert not args
            return _reduce_directly(expr, axis)
        return _reduce_directly(expr, axis, where, *args)

    doc_str = """Create a {0} expression over axis.

              Parameters
              ----------
              expr : Expr
                  The source expression.
              axis : IterVar
                  The reduction IterVar axis
              where : optional, Expr
                  Filtering predicate of the reduction.
              Returns
              -------
              value : Expr
                  The result value.

              Example
              -------
              .. code-block:: python

                m = tvm.var("m")
                n = tvm.var("n")
                A = tvm.placeholder((m, n), name="A")
                k = tvm.reduce_axis((0, n), name="k")

                # there are two way to use this {0} reducer:
                # mode 1, accept (expr, axis, where) to produce an Reduce Expr
                B = tvm.compute((m,), lambda i: tvm.{0}(A[i, k], axis=k), name="B")

                # mode 2, simply use it with multiple Exprs:
                {0}_res = tvm.{0}(m, n)
              """
    reducer.__doc__ = doc_str.format(name)
    return reducer


_init_api("tvm.api")
#pylint: disable=unnecessary-lambda
sum = comm_reducer(lambda x, y: x+y, lambda t: const(0, dtype=t), name="sum")
min = comm_reducer(lambda x, y: _make.Min(x, y), max_value, name='min')
max = comm_reducer(lambda x, y: _make.Max(x, y), min_value, name='max')
