
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

_CLASS_DTYPE = None

def _set_class_dtype(cls):
    global _CLASS_DTYPE
    _CLASS_DTYPE = cls


def _create_dtype_from_tuple(cls, code, bits, lanes):
    cdef DLDataType cdtype
    cdtype.code = code
    cdtype.bits = bits
    cdtype.lanes = lanes
    ret = cls.__new__(cls, str(cdtype))
    (<DataType>ret).cdtype = cdtype
    return ret


cdef class DataType:
    """DataType is a wrapper around DLDataType.

    Parameters
    ----------
    dtype_str : str
        The string representation of the data type
    """
    cdef DLDataType cdtype

    def __init__(self, dtype_str):
        cdef ByteArrayArg dtype_str_arg = ByteArrayArg(c_str(dtype_str))
        CHECK_CALL(TVMFFIDataTypeFromString(dtype_str_arg.cptr(), &(self.cdtype)))

    def __reduce__(self):
        cls = type(self)
        return (_create_dtype_from_tuple,
                (cls, self.cdtype.code, self.cdtype.bits, self.cdtype.lanes))

    def __eq__(self, other):
        if not isinstance(other, DataType):
            return False
        return (
            self.cdtype.code == other.cdtype.code
            and self.cdtype.bits == other.cdtype.bits
            and self.cdtype.lanes == other.cdtype.lanes
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def type_code(self):
        return self.cdtype.code

    @property
    def bits(self):
        return self.cdtype.bits

    @property
    def lanes(self):
        return self.cdtype.lanes

    @property
    def itemsize(self):
        """Get the number of bytes of a single element of this data type. When the number of lanes
        is greater than 1, the itemsize is the size of the vector type.

        Returns
        -------
        itemsize : int
            The number of bytes of a single element of this data type
        """
        lanes_as_int = self.cdtype.lanes
        if lanes_as_int < 0:
            raise ValueError("Cannot determine itemsize for scalable vector types")
        return (self.cdtype.bits * self.cdtype.lanes + 7) // 8

    def __str__(self):
        cdef TVMFFIObjectHandle dtype_str
        cdef TVMFFIByteArray* bytes
        CHECK_CALL(TVMFFIDataTypeToString(&(self.cdtype), &dtype_str))
        bytes = TVMFFIBytesGetByteArrayPtr(dtype_str)
        res = py_str(PyBytes_FromStringAndSize(bytes.data, bytes.size))
        CHECK_CALL(TVMFFIObjectFree(dtype_str))
        return res


cdef inline object make_ret_dtype(TVMFFIAny result):
    cdtype = DataType.__new__(DataType)
    (<DataType>cdtype).cdtype = result.v_dtype
    val = str.__new__(_CLASS_DTYPE, cdtype.__str__())
    val.__tvm_ffi_dtype__ = cdtype
    return val
