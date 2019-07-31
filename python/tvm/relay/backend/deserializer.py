# License .to the Apache Software Foundation (ASF) under one
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
# pylint: disable=invalid-name
"""
The Relay Virtual Machine deserializer.

Python interface for deserializing a Relay VM.
"""
from tvm import module
from tvm._ffi.runtime_ctypes import TVMByteArray
from . import _vm
from . import vm as rly_vm

def _create_deserializer(code, lib):
    """Create a deserializer object.

    Parameters
    ----------
    code : bytearray
        The serialized virtual machine code.

    lib : :py:class:`~tvm.module.Module`
        The serialized runtime module/library that contains the hardware
        dependent binary code.

    Returns
    -------
    ret : Deserializer
        The created virtual machine deserializer.
    """
    if isinstance(code, (bytes, str)):
        code = bytearray(code)
    elif not isinstance(code, (bytearray, TVMByteArray)):
        raise TypeError("vm is expected to be the type of bytearray or " +
                        "TVMByteArray, but received {}".format(type(code)))

    if not isinstance(lib, module.Module):
        raise TypeError("lib is expected to be the type of tvm.module.Module" +
                        ", but received {}".format(type(lib)))
    return _vm._Deserializer(code, lib)


class Deserializer:
    """Relay VM deserializer.

    Parameters
    ----------
    code : bytearray
        The serialized virtual machine code.

    lib : :py:class:`~tvm.module.Module`
        The serialized runtime module/library that contains the hardware
        dependent binary code.
    """
    def __init__(self, code, lib):
        self.mod = _create_deserializer(code, lib)
        self._deserialize = self.mod["deserialize"]

    def deserialize(self):
        """Deserialize the serialized bytecode into a Relay VM.

        Returns
        -------
        ret : VirtualMachine
            The deserialized Relay VM.
        """
        return rly_vm.VirtualMachine(self._deserialize())
