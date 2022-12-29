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
""" Operation class for computation declaration."""
import inspect

# pylint: disable=invalid-name
from numbers import Integral as _Integral
from typing import List, Optional

import tvm._ffi
import tvm.arith._ffi_api
import tvm.tir
import tvm.tir._ffi_api
from tvm._ffi.base import string_types
from tvm.ir import Array
from tvm.runtime import convert

from . import _ffi_api
from . import tag as _tag
from . import tensor as _tensor


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
    shape = (shape,) if isinstance(shape, tvm.tir.PrimExpr) else shape
    dtype = "float32" if dtype is None else dtype
    return _ffi_api.Placeholder(shape, dtype, name)


def compute(shape, fcompute, name="compute", tag="", attrs=None, varargs_names=None):
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

    tag: str, optional
        Additional tag information about the compute.

    attrs: dict, optional
        The additional auxiliary attributes about the compute.

    varargs_names: list, optional
        The names to use for each of the varargs. If not supplied, the varargs
        will be called i1, i2, ...

    Returns
    -------
    tensor: Tensor
        The created tensor
    """
    if _tag.TagScope.get_current() is not None:
        if tag != "":
            raise ValueError("nested tag is not allowed for now")
        tag = _tag.TagScope.get_current().tag
    shape = (shape,) if isinstance(shape, tvm.tir.PrimExpr) else shape
    # for python3
    shape = tuple([int(s) if isinstance(s, float) else s for s in shape])
    out_ndim = len(shape)

    argspec = inspect.getfullargspec(fcompute)
    if len(argspec.args) == 0 and argspec.varargs is None:
        arg_names = ["i%d" % i for i in range(out_ndim)]
    elif argspec.varargs is not None:
        # if there is a varargs, it takes the remaining dimensions of out_ndim
        num_remaining_args = out_ndim - len(argspec.args)
        if varargs_names is not None:
            if len(varargs_names) != num_remaining_args:
                raise RuntimeError(
                    f"Number of varargs ({num_remaining_args}) does not match number"
                    f"of varargs_names ({len(varargs_names)})"
                )
            arg_names = argspec.args + varargs_names
        else:
            arg_names = argspec.args + [f"i{i}" for i in range(out_ndim - len(argspec.args))]
    else:
        arg_names = argspec.args
        # if there are fewer args than out dimensions, the remaining dimensions
        # are implicitly broadcast
        out_ndim = len(arg_names)
    assert argspec.varkw is None, "Variable keyword arguments not supported in fcompute"
    assert argspec.defaults is None, "Default arguments not supported in fcompute"
    assert len(argspec.kwonlyargs) == 0, "Keyword arguments are not supported in fcompute"

    if out_ndim != len(arg_names):
        raise ValueError(
            "Number of args to fcompute does not match dimension, "
            "args=%d, dimension=%d" % (len(arg_names), out_ndim)
        )

    dim_var = [tvm.tir.IterVar((0, s), x, 0) for x, s in zip(arg_names, shape[:out_ndim])]
    body = fcompute(*[v.var for v in dim_var])

    if isinstance(body, _tensor.TensorIntrinCall):
        for i, s in enumerate(shape[out_ndim:]):
            var_name = "ax" + str(i)
            dim_var.append(tvm.tir.IterVar((0, s), var_name, 4))
        op_node = _ffi_api.TensorComputeOp(
            name,
            tag,
            dim_var,
            body.reduce_axis,
            out_ndim,
            body.intrin,
            body.tensors,
            body.regions,
            body.scalar_inputs,
        )
    else:
        if not isinstance(body, (list, tuple)):
            body = [body]
        body = convert(body)
        op_node = _ffi_api.ComputeOp(name, tag, attrs, dim_var, body)

    num = op_node.num_outputs
    outputs = tuple(op_node.output(i) for i in range(num))
    return outputs[0] if num == 1 else outputs


def scan(init, update, state_placeholder, inputs=None, name="scan", tag="", attrs=None):
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

    tag: str, optional
        Additonal tag information about the compute.

    attrs: dict, optional
        The additional auxiliary attributes about the compute.

    Returns
    -------
    tensor: Tensor or list of Tensors
        The created tensor or tuple of tensors contains multiple outputs.

    Example
    -------
    .. code-block:: python

      # The following code is equivalent to numpy.cumsum
      m = te.var("m")
      n = te.var("n")
      X = te.placeholder((m, n), name="X")
      s_state = te.placeholder((m, n))
      s_init = te.compute((1, n), lambda _, i: X[0, i])
      s_update = te.compute((m, n), lambda t, i: s_state[t-1, i] + X[t, i])
      res = tvm.te.scan(s_init, s_update, s_state, X)
    """
    if _tag.TagScope.get_current() is not None:
        if tag != "":
            raise ValueError("nested tag is not allowed for now")
        tag = _tag.TagScope.get_current().tag
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
    axis = tvm.tir.IterVar((init[0].shape[0], update[0].shape[0]), "%s.idx" % name, 3)
    op = _ffi_api.ScanOp(name, tag, attrs, axis, init, update, state_placeholder, inputs)
    res = [op.output(i) for i in range(len(update))]
    return res[0] if len(res) == 1 else res


def extern(
    shape,
    inputs,
    fcompute,
    name="extern",
    dtype=None,
    in_buffers=None,
    out_buffers=None,
    tag="",
    attrs=None,
):
    """Compute several tensors via an extern function.

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

             - **ins** (list of :any:`tvm.tir.Buffer`) - Placeholder for each inputs
             - **outs** (list of :any:`tvm.tir.Buffer`) - Placeholder for each outputs

             **Returns**

             - **stmt** (:any:`tvm.tir.Stmt`) - The statement that carries out array computation.

    name: str, optional
        The name hint of the tensor

    dtype: str or list of str, optional
        The data types of outputs,
        by default dtype will be same as inputs.

    in_buffers: tvm.tir.Buffer or list of tvm.tir.Buffer, optional
        Input buffers.

    out_buffers: tvm.tir.Buffer or list of tvm.tir.Buffer, optional
        Output buffers.


    tag: str, optional
        Additonal tag information about the compute.

    attrs: dict, optional
        The additional auxiliary attributes about the compute.

    Returns
    -------
    tensor: Tensor or list of Tensors
        The created tensor or tuple of tensors contains multiple outputs.

    Example
    -------
    In the code below, C is generated by calling external PackedFunc
    `tvm.contrib.cblas.matmul`

    .. code-block:: python

        A = te.placeholder((n, l), name="A")
        B = te.placeholder((l, m), name="B")
        C = te.extern((n, m), [A, B],
                       lambda ins, outs: tvm.tir.call_packed(
                          "tvm.contrib.cblas.matmul",
                            ins[0], ins[1], outs[0], 0, 0), name="C")
    """
    if _tag.TagScope.get_current() is not None:
        if tag != "":
            raise ValueError("nested tag is not allowed for now")
        tag = _tag.TagScope.get_current().tag
    shape = (shape,) if isinstance(shape, (tvm.tir.PrimExpr, _Integral)) else shape
    if shape == () or isinstance(shape[0], (tvm.tir.PrimExpr, _Integral)):
        shape = [shape]
    if in_buffers is not None:
        in_buffers = [in_buffers] if not isinstance(in_buffers, list) else in_buffers
        if len(inputs) != len(in_buffers):
            raise RuntimeError(
                "Number of inputs and in_buffers mismatch: %d vs %d."
                % (len(inputs), len(in_buffers))
            )
    if out_buffers is not None:
        out_buffers = [out_buffers] if not isinstance(out_buffers, list) else out_buffers
        if len(shape) != len(out_buffers):
            raise RuntimeError(
                "Number of outputs and out_buffers mismatch: %d vs %d."
                % (len(shape), len(out_buffers))
            )
    input_placeholders = in_buffers or []
    output_placeholders = out_buffers or []
    types = set()
    for t in inputs:
        if not isinstance(t, _tensor.Tensor):
            raise ValueError("expect inputs to be tensor")
        if in_buffers is None:
            input_placeholders.append(
                tvm.tir.decl_buffer(
                    t.shape, t.dtype, t.op.name, elem_offset=tvm.tir.Var("elem_offset", "int32")
                )
            )
        types.add(t.dtype)

    if dtype is None:
        if len(types) != 1:
            raise ValueError("Cannot infer output type, please provide dtype argument")
        infered_type = types.pop()
        dtype = [infered_type for _ in shape]
    if isinstance(dtype, str):
        dtype = [dtype]

    if out_buffers is None:
        for shp, dt in zip(shape, dtype):
            output_placeholders.append(
                tvm.tir.decl_buffer(shp, dt, name, elem_offset=tvm.tir.Var("elem_offset", "int32"))
            )
    body = fcompute(input_placeholders, output_placeholders)
    if isinstance(body, tvm.tir.PrimExpr):
        body = tvm.tir.Evaluate(body)
    if not isinstance(body, tvm.tir.Stmt):
        raise ValueError(
            "Function '{}' should return PrimExpr or Stmt, but it returned '{}'".format(
                fcompute.__name__, type(body)
            )
        )

    op = _ffi_api.ExternOp(name, tag, attrs, inputs, input_placeholders, output_placeholders, body)
    res = [op.output(i) for i in range(len(output_placeholders))]
    return res[0] if len(res) == 1 else res


def extern_primfunc(input_tensors: List[_tensor.Tensor], primfunc: tvm.tir.PrimFunc, **kwargs):
    """Compute tensors via a schedulable TIR PrimFunc

    Parameters
    ----------
    input_tensors: list of Tensor
        Input tensors that map to the corresponding primfunc input params.

    primfunc: PrimFunc
        The TIR PrimFunc

    Returns
    -------
    tensor: Tensor or list of Tensors
        The created tensor or tuple of tensors if it contains multiple outputs.

    Example
    -------
    In the code below, a TVMScript defined TIR PrimFunc is inlined into
    a TE ExternOp. Applying te.create_prim_func on this

    .. code-block:: python

        A = te.placeholder((128, 128), name="A")
        B = te.placeholder((128, 128), name="B")

        @T.prim_func
        def before_split(a: T.handle, b: T.handle) -> None:
            A = T.match_buffer(a, (128, 128))
            B = T.match_buffer(b, (128, 128))
            for i, j in T.grid(128, 128):
                with T.block("B"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj] * 2.0

        C = te.extern_primfunc([A, B], func)
    """

    # dt_access_map and primfunc.buffer_map are unordered, so use order from primfunc.params
    dt_access_map = tvm.arith._ffi_api.DomainTouchedAccessMap(primfunc)
    ordered_buffers = [primfunc.buffer_map[param] for param in primfunc.params]
    in_buffers = [buf for buf in ordered_buffers if len(dt_access_map[buf][0])]
    out_buffers = [buf for buf in ordered_buffers if len(dt_access_map[buf][1])]
    assert in_buffers, "PrimFunc has no input buffers"
    assert out_buffers, "PrimFunc has no output buffers"

    outputs = []
    inplace = []
    input_buffers = in_buffers
    for obuf in out_buffers:
        if obuf in in_buffers:
            inplace.append(obuf)
        else:
            outputs.append(obuf)

    if not outputs:
        iobuf = inplace.pop()
        input_buffers.remove(iobuf)
        outputs = [iobuf]

    assert len(input_buffers) == len(input_tensors), (
        "The number of provided input input_tensors does not match the number of ",
        "input buffers in the primfunc",
    )
    for tensor, buffer in zip(input_tensors, input_buffers):
        # TODO(csullivan): Can a stronger comparison between Tensor<>Buffer be made?
        assert len(tensor.shape) == len(buffer.shape)
        for d1, d2 in zip(tensor.shape, buffer.shape):
            assert d1 == d2, (
                "The input input_tensors provided do not match the input buffers in the ",
                "primfunc. Please check that the order of input te.Input_Tensors and the ",
                "order of the primfunc variables in the params list agree.",
            )
    output = extern(
        [buf.shape for buf in outputs],
        input_tensors,
        lambda ins, outs: primfunc.body,
        in_buffers=input_buffers,
        out_buffers=outputs,
        **kwargs,
    )
    return output


def var(name="tindex", dtype="int32", span=None):
    """Create a new variable with specified name and dtype

    Parameters
    ----------
    name : str
        The name

    dtype : str
        The data type

    span : Optional[Span]
        The location of this variable in the source.

    Returns
    -------
    var : Var
        The result symbolic variable.
    """
    return tvm.tir.Var(name, dtype, span)


def const(dtype="int32", span=None):
    """Create a new constant with specified name and dtype

    Parameters
    ----------
    name : str
        The name

    dtype : str
        The data type

    span : Optional[Span]
        The location of this variable in the source.

    Returns
    -------
    var : Var
        The result symbolic variable.
    """
    return tvm.tir.const(dtype, span)


def size_var(name="size", dtype="int32", span=None):
    """Create a new variable represents a tensor shape size, which is non-negative.

    Parameters
    ----------
    name : str
        The name

    dtype : str
        The data type

    span : Optional[Span]
        The location of this variable in the source.

    Returns
    -------
    var : SizeVar
        The result symbolic shape variable.
    """
    return tvm.tir.SizeVar(name, dtype, span)


def thread_axis(dom=None, tag="", name="", span=None):
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

    span : Optional[Span]
        The location of this variable in the source.

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
    return tvm.tir.IterVar(dom, name, 1, tag, span)


def reduce_axis(dom, name="rv", thread_tag="", span=None):
    """Create a new IterVar for reduction.

    Parameters
    ----------
    dom : Range
        The domain of iteration.

    name : str
        The name of the variable.

    thread_tag : Optional[str]
        The name of the thread_tag.

    span : Optional[Span]
        The location of this variable in the source.

    Returns
    -------
    axis : IterVar
        An iteration variable representing the value.
    """
    return tvm.tir.IterVar(dom, name, 2, thread_tag, span)


def create_prim_func(
    ops: List[_tensor.Tensor], index_dtype_override: Optional[str] = None
) -> tvm.tir.PrimFunc:
    """Create a TensorIR PrimFunc from tensor expression

    Parameters
    ----------
    ops : List[Tensor]
        The source expression.

    Example
    -------
    We define a matmul kernel using following code:

    .. code-block:: python

        import tvm
        from tvm import te
        from tvm.te import create_prim_func
        import tvm.script

        A = te.placeholder((128, 128), name="A")
        B = te.placeholder((128, 128), name="B")
        k = te.reduce_axis((0, 128), "k")
        C = te.compute((128, 128), lambda x, y: te.sum(A[x, k] * B[y, k], axis=k), name="C")
        func = create_prim_func([A, B, C])
        print(func.script())

    If we want to use TensorIR schedule to do transformations on such kernel,
    we need to use `create_prim_func([A, B, C])` to create a schedulable PrimFunc.
    The generated function looks like:

    .. code-block:: python

        @T.prim_func
        def tir_matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
            A = T.match_buffer(a, (128, 128))
            B = T.match_buffer(b, (128, 128))
            C = T.match_buffer(c, (128, 128))

            for i, j, k in T.grip(128, 128, 128):
                with T.block():
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = 0.0
                    C[vi, vj] += A[vi, vk] * B[vj, vk]

    Returns
    -------
    func : tir.PrimFunc
        The created function.
    """
    if not isinstance(ops, (list, tuple, Array)):
        ops = [ops]
    return _ffi_api.CreatePrimFunc(ops, index_dtype_override)
