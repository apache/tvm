"""Tensor and Operation class for computation declaration."""
# pylint: disable=invalid-name
from __future__ import absolute_import as _abs
import numpy as _np
from .._ffi.node import register_node
from .. import expr as _expr
from .. import api as _api
from .. import tensor as _tensor
from .. import ndarray as _nd

float32 = "float32"
csr = "csr"

@register_node
class CSRNDArray(object):
    """Sparse tensor object in CSR format."""
    def __init__(self, arg1, ctx=None, shape=None):
        """Construct a sparse matrix in CSR format.

        Parameters
        ----------
        arg1 : numpy.ndarray or a tuple with (data, indices, indptr)
            The corresponding a dense numpy array,
            or a tuple for constructing a sparse matrix directly.

        ctx: tvm.TVMContext
            The corresponding context.
        """
        if isinstance(arg1, tuple):
            self.data, self.indices, self.indptr = arg1[0], arg1[1], arg1[2]
            self.shape = shape
        elif isinstance(arg1, _np.ndarray):
            source_array = arg1
            ridx, cidx = _np.nonzero(source_array)
            data = source_array[ridx, cidx]
            self.data = _nd.array(data, ctx)
            indices = _np.nonzero(source_array)[1].astype('int32')
            self.indices = _nd.array(indices, ctx)
            indptr = [0]+_np.apply_along_axis(_np.count_nonzero, axis=1, arr=source_array).tolist()
            indptr = _np.cumsum(_np.array(indptr, 'int32')).astype('int32')
            self.indptr = _nd.array(indptr, ctx)
            self.shape = source_array.shape
        else:
            raise RuntimeError("Construct CSRNDArray with either a tuple (data, indices, indptr) "
                               "or a numpy.array, can't handle type %s." % (type(arg1),))
        self.stype = 'csr'
        self.dtype = self.data.dtype
        assert self.shape is not None
        assert isinstance(self.data, _nd.NDArray)
        assert isinstance(self.indices, _nd.NDArray)
        assert str(self.indices.dtype) == 'int32' or \
            str(self.indices.dtype) == 'int64', str(self.indices.dtype)
        assert isinstance(self.indptr, _nd.NDArray)
        assert str(self.indptr.dtype) == 'int32' or \
            str(self.indptr.dtype) == 'int64', str(self.indptr.dtype)

    def asnumpy(self):
        """Construct a full matrix and convert it to numpy array."""
        full = _np.zeros(self.shape, self.dtype)
        ridx = _np.diff(self.indptr.asnumpy())
        ridx = _np.hstack((_np.ones((v,), 'int32')*i for i, v in enumerate(ridx)))
        full[ridx, self.indices.asnumpy().astype('int32')] = self.data.asnumpy()
        return full

def array(source_array, ctx=None, shape=None):
    """Construct a CSRNDArray from numpy.ndarray"""
    return CSRNDArray(source_array, shape=shape, ctx=ctx)

@register_node
class CSRPlaceholderOp(object):
    """Placeholder class for CSR based sparse tensor representation."""
    def __init__(self, shape, nonzeros, dtype, name, stype):
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
        self.shape = shape
        self.dtype = dtype
        self.name = name
        self.stype = stype
        self.data = _api.placeholder((nonzeros,), dtype=dtype, name=self.name+'_data')
        self.indices = _api.placeholder((nonzeros,), dtype='int32', name=self.name+'_indices')
        self.indptr = _api.placeholder((self.shape[0]+1,), dtype='int32', name=self.name+'_indptr')
        assert isinstance(self.data, _tensor.Tensor)
        assert isinstance(self.indices, _tensor.Tensor)
        assert isinstance(self.indptr, _tensor.Tensor)

def placeholder(shape, nonzeros=None, dtype=None, name="placeholder", stype=None):
    """Construct an empty sparse tensor object.

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
    tensor: CSRNDArray
        The created tensor
    """
    shape = (shape,) if isinstance(shape, _expr.Expr) else shape
    nonzeros = 0 if nonzeros is None else nonzeros
    dtype = float32 if dtype is None else dtype
    stype = csr if stype is None else stype
    return CSRPlaceholderOp(shape=shape, nonzeros=nonzeros, dtype=dtype, name=name, stype=stype)
