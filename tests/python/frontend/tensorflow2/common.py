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
# pylint: disable=import-self, invalid-name, unused-argument, too-many-lines, len-as-condition, broad-except
# pylint: disable=import-outside-toplevel, redefined-builtin
"""TF2 to relay converter test utilities"""

import tvm
from tvm import relay

from tvm.runtime.vm import VirtualMachine
import tvm.contrib.graph_executor as runtime
from tvm.relay.frontend.tensorflow import from_tensorflow
import tvm.testing

import tensorflow as tf
from tensorflow.python.eager.def_function import Function

def vmobj_to_list(o):
    if isinstance(o, tvm.nd.NDArray):
        out = o.asnumpy().tolist()
    elif isinstance(o, tvm.runtime.container.ADT):
        result = []
        for f in o:
            result.append(vmobj_to_list(f))
        out = result
    else:
        raise RuntimeError("Unknown object type: %s" % type(o))
    return out

def run_tf_code(func, input_):
    if type(func) is Function:
        out = func(input_)
        if isinstance(out, list):
            a = [x.numpy() for x in out]
        else:
            a = out.numpy()
    else:
        a = func(tf.constant(input_))
        if type(a) is dict:
            a = [x.numpy() for x in a.values()]
            if len(a) == 1:
                a = a[0]
        elif type(a) is list:
            a = [x.numpy() for x in a]
            if len(a) == 1:
                a = a[0]
        else:
            a = a.numpy()
    return a

def compile_graph_runtime(mod, params, target = "llvm", target_host = "llvm",
            opt_level=3, output_sig=None):
    with tvm.transform.PassContext(opt_level):
        lib = relay.build(mod, target=target, target_host=target_host, params=params)
    return lib

def compile_vm(mod, params, target = "llvm", target_host = "llvm",
               opt_level=3, disabled_pass=None, output_sig=None):
    with tvm.transform.PassContext(opt_level, disabled_pass=disabled_pass):
        mod = relay.transform.InferType()(mod)
        vm_exec = relay.vm.compile(mod, target, target_host, params=params)
    return vm_exec

def run_vm(vm_exec, input_, ctx = tvm.cpu(0)):
    vm = VirtualMachine(vm_exec, ctx)
    _out = vm.invoke("main", input_)
    return vmobj_to_list(_out)

def run_graph(lib, input_, ctx = tvm.cpu(0)):
    mod = runtime.GraphModule(lib["default"](ctx))
    mod.set_input(0, input_)
    mod.run()
    _out = mod.get_output(0).asnumpy()
    return _out

def compare_tf_tvm(gdef, input_, output_, vm=True, output_sig=None):
    """ compare tf and tvm execution for the same input.

    Parameters
    ----------
    func: tf function. can be from saved model or not. different ways to pass input
        from saved model: <class 'tensorflow.python.saved_model.load._WrapperFunction'>
        not from saved model:  <class 'tensorflow.python.eager.def_function.Function'>

    mod: compiled relay module (vm or graph runtime). converted from tf func.

    input_: a single numpy array object

    """
    mod, params = from_tensorflow(gdef, outputs=output_sig)
    if vm:
        exec_ = compile_vm(mod, params, output_sig=output_sig)
        tvm_out = run_vm(exec_, input_)
    else:
        lib = compile_graph_runtime(mod, params, output_sig=output_sig)
        tvm_out = run_graph(lib, input_)
    tvm.testing.assert_allclose(output_, tvm_out, atol=1e-5)
