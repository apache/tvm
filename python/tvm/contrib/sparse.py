"""Tensor and Operation class for computation declaration."""
# pylint: disable=invalid-name
from __future__ import absolute_import as _abs
from .._ffi.node import NodeBase, NodeGeneric, register_node, convert_to_node
from .. import _api_internal
from .. import make as _make
from .. import expr as _expr
from .. import api as _api
from .. import tag as _tag
from .. import tensor as _tensor
from .. import schedule as _schedule

float32 = "float32"
csr = "csr"

@register_node
class CSRNDArray(object):
    """Tensor object, to construct, see function.Tensor"""
    def __init__(self, shape, dtype='float32', name='',
                 data=None, indices=None, indptr=None):
        self.stype = 'csr'
        self.shape = shape
        self.dtype = dtype
        self.name = name
        if data is None:
            self.data = _api.placeholder(shape, dtype, name+'_data')
        else:
            self.data = data
        if indices is None:
            self.indices = _api.placeholder(shape, 'int32', name+'_indices')
        else:
            self.indices = indices
        if indptr is None:
            self.indptr = _api.placeholder(shape, 'int32', name+'_indptr')
        else:
            self.indptr = indptr
        assert isinstance(self.data, _tensor.Tensor)
        assert isinstance(self.indices, _tensor.Tensor)
        assert isinstance(self.indptr, _tensor.Tensor)

def array(source_array, ctx=None, dtype=None):
    ret = None
    import numpy
    if isinstance(source_array, numpy.ndarray):
        return CSRNDArray(shape=source_array.shape, dtype=str(source_array.dtype))
    return ret

@register_node
class CSRPlaceholderOp(_tensor.Operation):
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
        super(CSRPlaceholderOp, self).__init__(self)
        self.shape = shape
        self.dtype = dtype
        self.name = name
        self.stype = stype
        shape = (0,)
        self.data = _api.placeholder(shape, dtype, name+'_data')
        self.indices = _api.placeholder(shape, 'int32', name+'_indices')
        self.indptr = _api.placeholder(shape, 'int32', name+'_indptr')

# 
# @register_node
# class CSRBuffer(_schedule.Buffer):
#     """Placeholder class for csr based sparse tensor representation."""
#     def __init__(self, shape, dtype, name, stype):
#         """Contructing a bare bone structure for a csr_matrix
# 
#         Parameters
#         ----------
#         shape: Tuple of Expr
#             The shape of the tensor
# 
#         dtype: str, optional
#             The data type of the tensor
# 
#         name: str, optional
#             The name hint of the tensor
# 
#         stype: str, optional
#             The storage type of the tensor
#         """
#         super(CSRBuffer, self).__init__(self)
#         self.shape = shape
#         self.dtype = dtype
#         self.name = name
#         self.stype = stype
#         shape = (0,)
#         self.data = _api.decl_buffer(shape, dtype, name+'_data')
#         self.indices = _api.decl_buffer(shape, 'int32', name+'_indices')
#         self.indptr = _api.decl_buffer(shape, 'int32', name+'_indptr')
# 

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
