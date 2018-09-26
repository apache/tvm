"""Tensor intrinsics"""
from __future__ import absolute_import as _abs
from . import _api_internal
from . import api as _api
from . import expr as _expr
from . import stmt as _stmt
from . import make as _make
from . import tensor as _tensor
from . import schedule as _schedule
from .build_module import current_build_config
from ._ffi.node import NodeBase, register_node


def _get_region(tslice):
    region = []
    for idx in tslice.indices:
        if isinstance(idx, slice):
            assert idx.step is None
            region.append(_api.Range(idx.start, idx.stop))
        else:
            if isinstance(idx, _schedule.IterVar):
                begin = idx.var
            else:
                begin = idx
            region.append(_make.range_by_min_extent(begin, 1))
    return region

@register_node
class TensorIntrin(NodeBase):
    """Tensor intrinsic functions for certain computation.

    See Also
    --------
    decl_tensor_intrin: Construct a TensorIntrin
    """
    def __call__(self, *args, **kwargs):
        tensors = [x.tensor for x in args]
        regions = [_get_region(x) for x in args]
        reduce_axis = []
        if "reduce_axis" in kwargs:
            reduce_axis = kwargs["reduce_axis"]
            if not isinstance(reduce_axis, (list, tuple)):
                reduce_axis = [reduce_axis]
            reduce_axis = _api.convert(reduce_axis)
        return _api_internal._TensorIntrinCall(self, tensors, regions, reduce_axis)

def decl_tensor_intrin(op,
                       fcompute,
                       name="tensor_intrin",
                       binds=None):
    """Declare a tensor intrinsic function.

    Parameters
    ----------
    op: Operation
        The symbolic description of the intrinsic operation

    fcompute: lambda function of inputs, outputs-> stmt
        Specifies the IR statement to do the computation.
        See the following note for function signature of fcompute

        .. note::
             **Parameters**

             - **ins** (list of :any:`Buffer`) - Placeholder for each inputs
             - **outs** (list of :any:`Buffer`) - Placeholder for each outputs

             **Returns**

             - **stmt** (:any:`Stmt`, or tuple of three stmts)
             - If a single stmt is returned, it represents the body
             - If tuple of three stmts are returned they corresponds to body,
               reduce_init, reduce_update

    name: str, optional
        The name of the intrinsic.

    binds: dict of :any:`Tensor` to :any:`Buffer`, optional
        Dictionary that maps the Tensor to Buffer which specified the data layout
        requirement of the function. By default, a new compact buffer is created
        for each tensor in the argument.

    Returns
    -------
    intrin: TensorIntrin
        A TensorIntrin that can be used in tensorize schedule.
    """
    if not isinstance(op, _tensor.Operation):
        raise TypeError("expect Operation")
    inputs = op.input_tensors
    binds = binds if binds else {}
    tensors = [x for x in inputs]
    for i in range(op.num_outputs):
        tensors.append(op.output(i))

    binds_list = []
    for t in inputs:
        if not isinstance(t.op, _tensor.PlaceholderOp):
            raise ValueError("Do not yet support composition op")

    cfg = current_build_config()
    for t in tensors:
        buf = (binds[t] if t in binds else
               _api.decl_buffer(t.shape, t.dtype, t.op.name,
                                data_alignment=cfg.data_alignment,
                                offset_factor=cfg.offset_factor))
        binds_list.append(buf)

    body = fcompute(binds_list[:len(inputs)], binds_list[len(inputs):])
    if isinstance(body, (_expr.Expr, _stmt.Stmt)):
        body = [body]
    body = [_make.Evaluate(x) if isinstance(x, _expr.Expr) else x for x in body]
    if len(body) < 3:
        body += [None] * (3 - len(body))
    return _api_internal._TensorIntrin(
        name, op, inputs, binds_list, *body)
