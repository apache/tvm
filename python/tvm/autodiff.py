"""Namespace of autodiff-related functions.

The functions are automatically exported from C++ side via PackedFunc.
You can read "include/tvm/autodiff.h" for the function signature of these functions.
"""
import logging

from ._ffi.function import _init_api
from ._ffi.node import NodeBase, register_node

_init_api("tvm.autodiff")

@register_node
class DifferentiationResult(NodeBase):
    """Result of differentiation.

    Parameters
    ----------
    result : list of Tensor
        The requested adjoints, i.e. the jacobians or gradients of the given output
        wrt to the given inputs.

    adjoints : dict from Tensor to Tensor
        A map from tensors to the corresponding adjoints (including internal nodes).

    adjoint_summands : dict from Tensor to dict from Tensor to Tensor
        Single summands of the adjoints.
    """
    def __getattr__(self, name):
        # Here we convert tvm Maps to dicts because Map compares keys by reference which is
        # wrong for Tensors. Hopefully, in the future Map gets fixed somehow, and this function
        # may be removed then.
        res = NodeBase.__getattr__(self, name)
        if name == 'adjoints':
            return dict(res.items())
        if name == 'adjoint_summands':
            return {k: dict(v.items()) for k, v in res.items()}
        return res

    def __getitem__(self, i):
        return self.result[i]

    def __len__(self):
        return len(self.result)


def differentiate(output, inputs=None, head=None, manual=None, fdiff=None):
    """Perform reverse-mode automatic differentiation.

    Example::

        x = tvm.placeholder((32, 3, 28, 28), name='x')
        w1 = tvm.placeholder((10, 3, 3, 3), name='w1')
        w2 = tvm.placeholder((10, 10, 3, 3), name='w2')
        y = topi.sum(topi.nn.conv2d(topi.nn.conv2d(x, w1, 1, 0), w2, 1, 0))

        [dw1, dw2] = tvm.differentiate(y, [w1, w2])

    Parameters
    ----------
    output : Tensor
        The tensor to differentiate.

    inputs : list of Tensor
        The list of input tensors. When the list is empty or None, will perform
        differentiation wrt all tensors the output depends on (i.e. will compute all
        adjoints and populate the corresponding dict, but the list of results
        will be empty).

    head : Tensor
        The adjoint of the output, in other words, some tensor, by which the Jacobians
        will be multiplied. Its shape must be of the form `prefix + output.shape`.
        If `None` is passed, the identity tensor of shape `output.shape + output.shape`
        will be used.

    manual : dict (Tensor, Tensor) -> function
        A dict providing custom multiplication-differentiation functions (see `fdiff`)
        for certain pairs of tensors. Each pair consists of an output and an input tensor,
        the input one being an immediate dependency of the output one. Pairs of the form
        `(None, tensor)` and `(tensor, None)` are allowed, `None` working as a wildcard.

    fdiff : function (Tensor, Tensor, Tensor) -> Tensor
        The default function performing differentiation and multiplication, by default
        `tvm.autodiff.FDiffBuildingBlock` is used. The function must accept three
        parameters:
        - `output` - an output tensor
        - `input` - an input tensor
        - `head` - the adjoint of the output tensor
        The result should be `head` multiplied by the jacobian of `output` wrt `input`

    Returns
    -------
    differentiation_result : DifferentiationResult
    """
    if inputs is None:
        inputs = []

    if fdiff is None:
        fdiff = DiffBuildingBlock

    if manual is not None:
        if not isinstance(manual, dict):
            manual = dict(manual)

        # pylint: disable=dangerous-default-value
        used_items = set()

        def _modified_fdiff(out, inp, head, manual=manual, old_fdiff=fdiff, used_items=used_items):
            if (out, inp) in manual:
                used_items.add((out, inp))
                return manual[(out, inp)](out, inp, head)
            if (out, None) in manual:
                used_items.add((out, None))
                return manual[(out, None)](out, inp, head)
            if (None, inp) in manual:
                used_items.add((None, inp))
                return manual[(None, inp)](out, inp, head)
            return old_fdiff(out, inp, head)

        fdiff = _modified_fdiff

    res = Differentiate(output, inputs, head, fdiff)

    if manual is not None:
        for k in manual:
            if k not in used_items:
                logging.warning("The manually specified differentiation function "
                                "for %s hasn't been used", k)

    return res
