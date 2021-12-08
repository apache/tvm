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
# pylint: disable=invalid-name
from numbers import Integral as _Integral
from typing import List, Union

import tvm._ffi
from tvm._ffi.base import string_types
from tvm.ir import Array
from tvm.runtime import convert
import tvm.tir
import tvm.tir._ffi_api

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


def compute(shape, fcompute, name="compute", tag="", attrs=None):
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
    ndim = len(shape)
    code = fcompute.__code__

    out_ndim = ndim
    if code.co_argcount == 0:
        arg_names = ["i%d" % i for i in range(ndim)]
    else:
        arg_names = code.co_varnames[: code.co_argcount]
        out_ndim = code.co_argcount

    if out_ndim != len(arg_names):
        raise ValueError("fcompute do not match dimension, ndim=%d" % ndim)

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
        The created tensor or tuple of tensors it it contains multiple outputs.

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
        The created tensor or tuple of tensors it it contains multiple outputs.

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
            input_placeholders.append(tvm.tir.decl_buffer(t.shape, t.dtype, t.op.name))
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
            output_placeholders.append(tvm.tir.decl_buffer(shp, dt, name))
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


def create_prim_func(ops: List[_tensor.Tensor]) -> tvm.tir.PrimFunc:
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
    return _ffi_api.CreatePrimFunc(ops)


def create_prim_func_from_outputs(
    outputs: Union[_tensor.Tensor, List[_tensor.Tensor]],
) -> tvm.tir.PrimFunc:
    """Create a TensorIR PrimFunc from output tensor(s) in TE

    Parameters
    ----------
    outputs : Union[Tensor, List[Tensor]]
        The source expression.

    Returns
    -------
    func : tir.PrimFunc
        The created function.
    """
    if not isinstance(outputs, (list, tuple, Array)):
        outputs = [outputs]
    return _ffi_api.CreatePrimFuncFromOutputs(outputs)
