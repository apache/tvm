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

from ._ffi.function import _init_api
from ._ffi.object import ObjectBase, ObjectTag, register_object

_init_api("tvm.object", __name__)

@register_object
class TensorObject(ObjectBase):
    """Tensor object."""
    tag = ObjectTag.TENSOR

    def __init__(self, handle):
        """Constructs a tensor object

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
        self.data = GetTensorData(self)

    def asnumpy(self):
        """Convert data to numpy array
n
        Returns
        -------
        np_arr : numpy.ndarray
            The corresponding numpy array.
        """
        return self.data.asnumpy()


@register_object
class DatatypeObject(ObjectBase):
    """Datatype object."""
    tag = ObjectTag.DATATYPE

    def __init__(self, handle):
        """Constructs a tensor object

        Parameters
        ----------
        handle : object
            Object handle

        Returns
        -------
        obj : TensorObject
            A tensor object.
        """
        super(DatatypeObject, self).__init__(handle)
        self.tag = GetDatatypeTag(self)
        self.num_fields = GetDatatypeNumberOfFields(self)

    def __getitem__(self, idx):
        idx = idx + self.num_fields if idx < 0 else idx
        assert 0 <= idx < self.num_fields
        return GetDatatypeFields(self, idx)

    def __len__(self):
        return self.num_fields
