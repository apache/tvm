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
The Relay Virtual Machine serializer.

Python interface for serializing a Relay VM.
"""
import tvm
from . import _vm
from . import vm as rly_vm

def _create_serializer(executable):
    """Create a VM serializer.

    Parameters
    ----------
    executable : Union[Executable, :py:class:`~tvm.module.Module`]
        The virtual machine executable to be serialized.

    Returns
    -------
    ret : Serializer
        The created virtual machine executable serializer.
    """
    if isinstance(executable, rly_vm.Executable):
        executable = executable.module
    elif not isinstance(executable, tvm.module.Module):
        raise TypeError("executable is expected to be an Executable or " +
                        "tvm.Module, but received {}".format(type(executable)))

    return _vm._Serializer(executable)


class Serializer:
    """Relay VM serializer."""
    def __init__(self, executable):
        self.mod = _create_serializer(executable)
        self._get_lib = self.mod["get_lib"]
        self._serialize = self.mod["serialize"]

    def serialize(self):
        """Serialize the Relay VM.

        Returns
        -------
        code : bytearray
            The binary blob representing a serialized Relay VM. It can then be
            saved to disk and later deserialized into a new VM.

        lib : :py:class:`~tvm.module.Module`
            The runtime module that contains the generated code. It is
            basically a library that is composed of hardware dependent code.

        Notes
        -----
        The returned code is organized with the following sections in order.
         - Global section. This section contains the globals used by the
         virtual machine.
         - Constant section. This section is used to store the constant pool of
         a virtual machine.
         - Primitive name section. This section is introduced to accommodate
         the list of primitive operator names that will be invoked by the
         virtual machine.
         - Code section. The VM functions, including bytecode, are sitting in
         this section.

        Examples
        --------
        .. code-block:: python

            import numpy as np
            import tvm
            from tvm import relay

            # define a simple network.
            x = relay.var('x', shape=(10, 10))
            f = relay.Function([x], x + x)
            mod = relay.Module({"main": f})

            # create a Relay VM.
            ctx = tvm.cpu()
            target = "llvm"
            executable = relay.vm..compile(mod, target)
            executable.set_context(ctx)

            # serialize.
            ser = relay.serializer.Serializer(executable)
            code, lib = ser.serialize()

            # save and load the code and lib file.
            tmp = tvm.contrib.util.tempdir()
            path_lib = tmp.relpath("lib.so")
            lib.export_library(path_lib)
            with open(tmp.relpath("code.ro"), "wb") as fo:
                fo.write(code)

            loaded_lib = tvm.module.load(path_lib)
            loaded_code = bytearray(open(tmp.relpath("code.ro"), "rb").read())

            # deserialize.
            deser = relay.deserializer.Deserializer(loaded_code, loaded_lib)
            des_exec = deser.deserialize()

            # execute the deserialized executable.
            des_exec.set_context(ctx)
            x_data = np.random.rand(10, 10).astype('float32')
            des_vm = relay.vm.VirtualMachine(des_exec)
            res = des_vm.run(x_data)
            print(res.asnumpy())
        """
        return self._serialize(), self._get_lib()
