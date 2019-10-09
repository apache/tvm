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
"""TVM Runtime Object API."""
from __future__ import absolute_import as _abs
import numpy as _np

from tvm._ffi.vmobj import Object, ObjectTag, register_object
from tvm import ndarray as _nd
from . import _vmobj

# TODO(@icemelon9): Add ClosureObject

@register_object
class TensorObject(Object):
    """Tensor object."""
    tag = ObjectTag.TENSOR

    def __init__(self, handle):
        """Constructs a Tensor object

        Parameters
        ----------
        handle : object
            Object handle

        Returns
        -------
        obj : TensorObject
            A tensor object.
        """
        super(TensorObject, self).__init__(handle)
        self.data = _vmobj.GetTensorData(self)

    def asnumpy(self):
        """Convert data to numpy array

        Returns
        -------
        np_arr : numpy.ndarray
            The corresponding numpy array.
        """
        return self.data.asnumpy()


@register_object
class DatatypeObject(Object):
    """Datatype object."""
    tag = ObjectTag.DATATYPE

    def __init__(self, handle):
        """Constructs a Datatype object

        Parameters
        ----------
        handle : object
            Object handle

        Returns
        -------
        obj : DatatypeObject
            A Datatype object.
        """
        super(DatatypeObject, self).__init__(handle)
        self.tag = _vmobj.GetDatatypeTag(self)
        num_fields = _vmobj.GetDatatypeNumberOfFields(self)
        self.fields = []
        for i in range(num_fields):
            self.fields.append(_vmobj.GetDatatypeFields(self, i))

    def __getitem__(self, idx):
        return self.fields[idx]

    def __len__(self):
        return len(self.fields)

    def __iter__(self):
        return iter(self.fields)

# TODO(icemelon9): Add closure object

def tensor_object(arr, ctx=_nd.cpu(0)):
    """Create a tensor object from source arr.

    Parameters
    ----------
    arr : numpy.ndarray or tvm.nd.NDArray
        The source array.

    ctx :  TVMContext, optional
        The device context to create the array

    Returns
    -------
    ret : TensorObject
        The created object.
    """
    if isinstance(arr, _np.ndarray):
        tensor = _vmobj.Tensor(_nd.array(arr, ctx))
    elif isinstance(arr, _nd.NDArray):
        tensor = _vmobj.Tensor(arr)
    else:
        raise RuntimeError("Unsupported type for tensor object.")
    return tensor


def tuple_object(fields):
    """Create a datatype object from source tuple.

    Parameters
    ----------
    fields : list[Object] or tuple[Object]
        The source tuple.

    Returns
    -------
    ret : DatatypeObject
        The created object.
    """
    for f in fields:
        assert isinstance(f, Object)
    return _vmobj.Tuple(*fields)


def datatype_object(tag, fields):
    """Create a datatype object from tag and source fields.

    Parameters
    ----------
    tag : int
        The tag of datatype.

    fields : list[Object] or tuple[Object]
        The source tuple.

    Returns
    -------
    ret : DatatypeObject
        The created object.
    """
    for f in fields:
        assert isinstance(f, Object)
    return _vmobj.Datatype(tag, *fields)
