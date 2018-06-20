"""Tensor and Operation class for computation declaration."""
# pylint: disable=invalid-name
from __future__ import absolute_import as _abs
from .._ffi.node import NodeBase, NodeGeneric, register_node, convert_to_node
from .. import _api_internal
from .. import make as _make
from .. import expr as _expr
from .. import api as _api

class TensorSlice(NodeGeneric, _expr.ExprOp):
    """Auxiliary data structure for enable slicing syntax from tensor."""
    def __init__(self, tensor, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        self.tensor = tensor
        self.indices = indices

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        return TensorSlice(self.tensor, self.indices + indices)

    def asnode(self):
        """Convert slice to node."""
        return self.tensor(*self.indices)

    @property
    def dtype(self):
        """Data content of the tensor."""
        return self.tensor.dtype

itervar_cls = None

@register_node
class CSRTensor(NodeBase, _expr.ExprOp):
    """Tensor object, to construct, see function.Tensor"""
    def __call__(self, *indices):
        ndim = self.ndim
        if len(indices) != ndim:
            raise ValueError("Need to provide %d index in tensor slice" % ndim)
        indices = convert_to_node(indices)
        args = []
        for x in indices:
            if isinstance(x, _expr.Expr):
                args.append(x)
            elif isinstance(x, iter_var_cls):
                args.append(x.var)
            else:
                raise ValueError("The indices must be expression")

        return _make.Call(self.dtype, self.op.name,
                          args, _expr.Call.Halide,
                          self.op, self.value_index)

    def __getitem__(self, indices):
        return TensorSlice(self, indices)

    def __hash__(self):
        return _api_internal._TensorHash(self)

    def __eq__(self, other):
        if not isinstance(other, Tensor):
            if isinstance(other, _expr.ExprOp):
                return _expr.EqualOp(self, other)
            return False
        if self.ndim == 0 and other.ndim == 0:
            raise ValueError("Equal == comparison among rank-0 tensor is ambiguous, "
                             "use Tensor.equal for content expression equvalence, "
                             "use Tensor.same_as for exact reference comparison")
        return _api_internal._TensorEqual(self, other)

    @property
    def ndim(self):
        """Dimension of the tensor."""
        return len(self.shape)

    @property
    def axis(self):
        """Axis of the tensor."""
        return self.__getattr__("axis")

    @property
    def op(self):
        """The corressponding :any:`Operation`."""
        return self.__getattr__("op")

    @property
    def value_index(self):
        """The output value index the tensor corressponds to."""
        return self.__getattr__("value_index")

    @property
    def shape(self):
        """The output shape of the tensor."""
        return self.__getattr__("shape")

    @property
    def name(self):
        op = self.op
        if op.num_outputs == 1:
            return op.name
        return "%s.v%d" % (op.name, self.value_index)


class Operation(NodeBase):
    """Represent an operation that generate a tensor"""
    def output(self, index):
        """Get the index-th output of the operation

        Parameters
        ----------
        index : int
            The index size.

        Returns
        -------
        out : Tensor
            The i-th output.
        """
        return _api_internal._OpGetOutput(self, index)

    @property
    def num_outputs(self):
        """Number of outputs of this op."""
        return _api_internal._OpNumOutputs(self)

    @property
    def input_tensors(self):
        """List of input tensors to this op."""
        return _api_internal._OpInputTensors(self)


@register_node
class PlaceholderOp(Operation):
    """Placeholder operation."""
    pass


@register_node
class ComputeOp(Operation):
    """Compute operation."""
    @property
    def axis(self):
        """Represent axis of IterVar, only defined when it is a ComputeOp"""
        return self.__getattr__("axis")

    @property
    def reduce_axis(self):
        """Represent axis of reductions, only defined when it is a ComputeOp"""
        return self.__getattr__("reduce_axis")


@register_node
class ScanOp(Operation):
    """Scan operation."""
    @property
    def scan_axis(self):
        """Represent axis of scan, only defined when it is a ScanOp"""
        return self.__getattr__("scan_axis")


@register_node
class ExternOp(Operation):
    """Extern operation."""
    pass

float32 = "float32"
csr = "csr"

class Placeholder(object):
    """Placeholder class for csr based sparse tensor representation."""
    def __init__(self, shape, dtype, name, stype):
        """Contructing a bare bone structure for a csr_matrix

        Parameters
        ----------
        shape: Tuple of Expr
            The shape of the tensor

        dtype: str, optional
            The data type of the tensor

        name: str, optional
            The name hint of the tensor

        stype: str, optional
            The storage type of the tensor
        """
        super(Placeholder, self).__init__()
        self.shape = shape
        self.dtype = dtype
        self.name = name
        self.stype = stype
        shape = (0,)
        self.data = _api.placeholder(shape, dtype, name+'_data')
        self.indices = _api.placeholder(shape, 'int32', name+'_indices')
        self.indptr = _api.placeholder(shape, 'int32', name+'_indptr')

def placeholder(shape, dtype=None, name="placeholder", stype=None):
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
    stype = csr if stype is None else stype
    return Placeholder(shape, dtype, name, stype)

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
