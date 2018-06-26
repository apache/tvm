"""Tensor and Operation class for computation declaration."""
# pylint: disable=invalid-name
from __future__ import absolute_import as _abs
import numpy as _np
from .._ffi.node import register_node
from .. import expr as _expr
from .. import api as _api
from .. import tensor as _tensor
from .. import schedule as _schedule
from .. import ndarray as _nd

float32 = "float32"
csr = "csr"

@register_node
class CSRNDArray(object):
    """Sparse tensor object in CSR format."""
    def __init__(self, source_array=None,
                 data=None, indices=None, indptr=None, ctx=None):
        """Construct a sparse matrix in CSR format."""
        self.stype = 'csr'
        self.shape = source_array.shape
        self.dtype = source_array.dtype
        if data is None:
            ridx, cidx = _np.nonzero(source_array)
            data = source_array[ridx, cidx]
            self.data = _nd.array(data, ctx)
        else:
            self.data = data
        if indices is None:
            indices = _np.nonzero(source_array)[1].astype('int32')
            self.indices = _nd.array(indices, ctx)
        else:
            self.indices = indices
        if indptr is None:
            indptr = [0]+_np.apply_along_axis(_np.count_nonzero, axis=1, arr=source_array).tolist()
            indptr = _np.cumsum(_np.array(indptr, 'int32')).astype('int32')
            self.indptr = _nd.array(indptr, ctx)
        else:
            self.indptr = indptr
        assert isinstance(self.data, _nd.NDArray)
        assert isinstance(self.indices, _nd.NDArray)
        assert str(self.indices.dtype) == 'int32', str(self.indices.dtype)
        assert isinstance(self.indptr, _nd.NDArray)
        assert str(self.indptr.dtype) == 'int32', str(self.indptr.dtype)

    def asnumpy(self):
        """Construct a full matrix and convert it to numpy array."""
        full = _np.zeros(self.shape, self.dtype)
        ridx = _np.diff(self.indptr.asnumpy())
        ridx = _np.hstack((_np.ones((v,), 'int32')*i for i, v in enumerate(ridx)))
        full[ridx, self.indices.asnumpy().astype('int32')] = self.data.asnumpy()
        return full

def array(source_array, ctx=None):
    """Construct a CSRNDArray from numpy.ndarray"""
    ret = None
    if isinstance(source_array, _np.ndarray):
        return CSRNDArray(source_array=source_array, ctx=ctx)
    return ret

@register_node
class CSRPlaceholderOp(object):
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
        self.shape = shape
        self.dtype = dtype
        self.name = name
        self.stype = stype
        self.data = _api.placeholder((shape[1],), dtype, name+'_data')
        self.indices = _api.placeholder((shape[1],), 'int32', name+'_indices')
        self.indptr = _api.placeholder((self.shape[0]+1,), 'int32', name+'_indptr')
        assert isinstance(self.data, _tensor.Tensor)
        assert isinstance(self.indices, _tensor.Tensor)
        assert isinstance(self.indptr, _tensor.Tensor)


@register_node
class CSRBuffer(_schedule.Buffer):
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
        super(CSRBuffer, self).__init__(self)
        self.shape = shape
        self.dtype = dtype
        self.name = name
        self.stype = stype
        shape = (0,)
        self.data = _api.decl_buffer(shape, dtype, name+'_data')
        self.indices = _api.decl_buffer(shape, 'int32', name+'_indices')
        self.indptr = _api.decl_buffer(shape, 'int32', name+'_indptr')


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
    tensor: CSRNDArray
        The created tensor
    """
    shape = (shape,) if isinstance(shape, _expr.Expr) else shape
    dtype = float32 if dtype is None else dtype
    stype = csr if stype is None else stype
    return CSRPlaceholderOp(shape, dtype, name, stype)
