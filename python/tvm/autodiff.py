"""Automatic differentiation of tensor expressions."""
from ._ffi.function import _init_api
from ._ffi.node import NodeBase, register_node

_init_api("tvm.autodiff")

@register_node
class DifferentiationResult(NodeBase):
    """Result of differentiation.

    Parameters
    ----------
    result : List[Tensor]
        The requested adjoints, i.e. the jacobians or gradients of the given output
        wrt to the given inputs.

    adjoints : Dict[Tensor, Tensor]
        A map from tensors to the corresponding adjoints (including internal nodes).

    adjoint_summands : Dict[Tensor, Dict[Tensor, Tensor]]
        Single summands of the adjoints.
    """

    # Here we convert tvm Maps to dicts because Map compares keys by reference which is
    # wrong for Tensors. Hopefully, in the future Map gets fixed somehow, and these properties
    # may be removed then.

    @property
    def adjoints(self):
        res = NodeBase.__getattr__(self, 'adjoints')
        return dict(res.items())

    @property
    def adjoint_summands(self):
        res = NodeBase.__getattr__(self, 'adjoint_summands')
        return {k: dict(v.items()) for k, v in res.items()}

    def _check_not_empty(self):
        if not self.result:
            raise ValueError("The result of differentiation does not contain any explicitly "
                             "requested results, so using it as an iterable is probably a mistake. "
                             "Please explicitly use res.adjoints to get adjoints or res.result to "
                             "get the empty list.")

    def __getitem__(self, i):
        self._check_not_empty()
        return self.result[i]

    def __len__(self):
        self._check_not_empty()
        return len(self.result)


def differentiate(output, inputs=None, head=None, override=None, fdiff=None):
    """Perform reverse-mode automatic differentiation.

    Parameters
    ----------
    output : Tensor
        The tensor to differentiate.

    inputs : List[Tensor]
        The list of input tensors. When the list is empty or None, will perform
        differentiation wrt all tensors the output depends on (i.e. will compute all
        adjoints and populate the corresponding dict, but the list of results
        will be empty).

    head : Tensor
        The adjoint of the output, in other words, some tensor, by which the Jacobians
        will be multiplied. Its shape must be of the form `prefix + output.shape`.
        If `None` is passed, the identity tensor of shape `output.shape + output.shape`
        will be used.

    override : Dict[Tensor, (List[Tensor], Callable[[Tensor, List[Tensor], Tensor], List[Tensor]])]
        Override differentiation for certain tensors. This dict maps tensors `t` to pairs
        `(dependencies, custom_diff)` where `dependencies` is a list of tensors which are considered
        to be inputs of `t` (which may differ from the immediate inputs), and `custom_diff` is a
        custom differentiation function which will be called as `custom_diff(t, dependencies,
        adjoint)` and should return a list of adjoints corresponding to dependencies. Note that this
        function differs from the one required for `fdiff` in that it takes a list of inputs instead
        of a single input and returns a list of adjoints instead of a single adjoint.

    fdiff : Callable[[Tensor, Tensor, Tensor], Tensor]
        The default function performing differentiation and multiplication, by default
        `tvm.autodiff.DiffBuildingBlock` is used. The function must accept three
        parameters:
        - `output` - an output tensor
        - `input` - an input tensor
        - `head` - the adjoint of the output tensor
        The result should be `head` multiplied by the jacobian of `output` wrt `input`

    Returns
    -------
    differentiation_result : DifferentiationResult

    Example
    -------
    .. code-block:: python

        x = tvm.placeholder((32, 3, 28, 28), name='x')
        w1 = tvm.placeholder((10, 3, 3, 3), name='w1')
        w2 = tvm.placeholder((10, 10, 3, 3), name='w2')
        z1 = topi.nn.conv2d(x, w1, 1, 1, 1)
        z2 = topi.nn.conv2d(z1, w2, 1, 1, 1)
        y = topi.sum(z2)

        # produce gradients
        [dw1, dw2] = tvm.differentiate(y, [w1, w2])

        # produce Jacobians
        [jw1, jw2] = tvm.differentiate(z2, [w1, w2])

        # produce gradients, the head adjoint for z2 is provided manually
        [dw1, dw2] = tvm.differentiate(z2, [w1, w2], topi.full_like(z2, 1.0))

        # produce gradients wrt all inputs
        res = tvm.differentiate(y)
        dw1 = res.adjoints[w1]
        dw2 = res.adjoints[w2]

        # a custom differentiation function
        def my_fdiff(out, inp, head):
            # this is the naive version, without any optimizations
            return topi.tensordot(head, tvm.autodiff.Jacobian(out, inp, False), len(out.shape))

        # using a custom differentiation function for everything
        [dw1, dw2] = tvm.differentiate(y, [w1, w2], fdiff=my_fdiff)

        # accessing individual summands of the adjoint
        y = z1 + z2
        res = tvm.differentiate(y, [w1, w2])
        [s1, s2] = res.adjoint_summands[z1].values()

        # a generalization of my_fdiff which works for non-immediate dependencies
        # this is necessary because z1 is not an immediate dep of z2 because of padding
        def my_diff(out, inputs, head):
            return tvm.differentiate(out, inputs, head, fdiff=my_fdiff)

        # using a custom differentiation function only for z2
        res = tvm.differentiate(y, [w1, w2], override={z2: ([z1, w2], my_diff)})
    """
    if inputs is None:
        inputs = []

    if fdiff is None:
        fdiff = DiffBuildingBlock

    if override is not None:
        # pylint: disable=dangerous-default-value
        def _modified_fdiff(out, inp, head, override=override, old_fdiff=fdiff, cache={}):
            if out in override:
                if (out, head) not in cache:
                    cache[(out, head)] = override[out][1](out, override[out][0], head)
                idx = override[out][0].index(inp)
                return cache[(out, head)][idx]
            return old_fdiff(out, inp, head)

        fdiff = _modified_fdiff

        override_deps = {t: deps for t, (deps, _) in override.items()}
        return Differentiate(output, inputs, head, fdiff, override_deps)
    return Differentiate(output, inputs, head, fdiff)
