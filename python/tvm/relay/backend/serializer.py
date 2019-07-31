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

def _create_serializer(vm):
    """Create a VM serializer.

    Parameters
    ----------
    vm : Union[VirtualMachine, :py:class:`~tvm.module.Module`]
        The virtual machine to be serialized.

    Returns
    -------
    ret : Serializer
        The created virtual machine serializer.
    """
    if isinstance(vm, rly_vm.VirtualMachine):
        vm = vm.module
    elif not isinstance(vm, tvm.module.Module):
        raise TypeError("vm is expected to be the type of VirtualMachine or " +
                        "tvm.Module, but received {}".format(type(vm)))

    return _vm._Serializer(vm)


class Serializer:
    """Relay VM serializer."""
    def __init__(self, vm):
        self.mod = _create_serializer(vm)
        self._get_lib = self.mod["get_lib"]
        self._get_bytecode = self.mod["get_bytecode"]
        self._get_globals = self.mod["get_globals"]
        self._get_stats = self.mod["get_stats"]
        self._get_primitive_ops = self.mod["get_primitive_ops"]
        self._serialize = self.mod["serialize"]

    @property
    def stats(self):
        """Get the statistics of the Relay VM.

        Returns
        -------
        ret : String
            The serialized statistic information.
        """
        return self._get_stats()

    @property
    def primitive_ops(self):
        """Get the name of the primitive ops that are executed in the VM.

        Returns
        -------
        ret : List[:py:class:`~tvm.expr.StringImm`]
            The list of primitive ops.
        """
        return [prim_op.value for prim_op in self._get_primitive_ops()]

    @property
    def bytecode(self):
        """Get the bytecode of the Relay VM.

        Returns
        -------
        ret : String
            The serialized bytecode.

        Notes
        -----
        The bytecode is in the following format:
          func_name reg_file_size num_instructions
          param1 param2 ... paramM
          instruction1
          instruction2
          ...
          instructionN

        Each instruction is printed in the following format:
          hash opcode field1 ... fieldX # The text format.

        The part starting from # is only used for visualization and debugging.
        The real serialized code doesn't contain it, therefore the deserializer
        doesn't need to deal with it as well.
        """
        return self._get_bytecode()

    @property
    def globals(self):
        """Get the globals used by the Relay VM.

        Returns
        -------
        ret : List[:py:class:`~tvm.expr.StringImm`]
            The serialized globals.
        """
        return [glb.value for glb in self._get_globals()]

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
            compiler = relay.vm.VMCompiler()
            vm = compiler.compile(mod, target)
            vm.init(ctx)

            # serialize.
            ser = relay.serializer.Serializer(vm)
            code, lib = ser.serialize()

            # save and load the code and lib file.
            tmp = tvm.contrib.util.tempdir()
            path_lib = tmp.relpath("lib.so")
            lib.export_library(path_lib)
            with open(tmp.relpath("code.bc"), "wb") as fo:
                fo.write(code)

            loaded_lib = tvm.module.load(path_lib)
            loaded_code = bytearray(open(tmp.relpath("code.bc"), "rb").read())

            # deserialize.
            deser = relay.deserializer.Deserializer(loaded_code, loaded_lib)
            des_vm = deser.deserialize()

            # execute the deserialized vm.
            des_vm.init(ctx)
            x_data = np.random.rand(10, 10).astype('float32')
            res = des_vm.run(x_data)
            print(res.asnumpy())
        """
        return self._serialize(), self._get_lib()
