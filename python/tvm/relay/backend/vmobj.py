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

from tvm._ffi.object import Object, register_object, getitem_helper
from tvm import ndarray as _nd
from . import _vmobj


@register_object("vm.Tensor")
class Tensor(Object):
    """Tensor object.

    Parameters
    ----------
    arr : numpy.ndarray or tvm.nd.NDArray
        The source array.

    ctx :  TVMContext, optional
        The device context to create the array
    """
    def __init__(self, arr, ctx=None):
        if isinstance(arr, _np.ndarray):
            ctx = ctx if ctx else _nd.cpu(0)
            self.__init_handle_by_constructor__(
                _vmobj.Tensor, _nd.array(arr, ctx=ctx))
        elif isinstance(arr, _nd.NDArray):
            self.__init_handle_by_constructor__(
                _vmobj.Tensor, arr)
        else:
            raise RuntimeError("Unsupported type for tensor object.")

    @property
    def data(self):
        return _vmobj.GetTensorData(self)

    def asnumpy(self):
        """Convert data to numpy array

        Returns
        -------
        np_arr : numpy.ndarray
            The corresponding numpy array.
        """
        return self.data.asnumpy()


@register_object("vm.ADT")
class ADT(Object):
    """Algebatic data type(ADT) object.

    Parameters
    ----------
    tag : int
        The tag of ADT.

    fields : list[Object] or tuple[Object]
        The source tuple.
    """
    def __init__(self, tag, fields):
        for f in fields:
            assert isinstance(f, Object)
        self.__init_handle_by_constructor__(
            _vmobj.ADT, tag, *fields)

    @property
    def tag(self):
        return _vmobj.GetADTTag(self)

    def __getitem__(self, idx):
        return getitem_helper(
            self, _vmobj.GetADTFields, len(self), idx)

    def __len__(self):
        return _vmobj.GetADTNumberOfFields(self)


def tuple_object(fields):
    """Create a ADT object from source tuple.

    Parameters
    ----------
    fields : list[Object] or tuple[Object]
        The source tuple.

    Returns
    -------
    ret : ADT
        The created object.
    """
    for f in fields:
        assert isinstance(f, Object)
    return _vmobj.Tuple(*fields)
