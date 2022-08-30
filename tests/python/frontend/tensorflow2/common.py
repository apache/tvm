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
from tvm.relay.frontend.tensorflow2 import from_tensorflow
import tvm.testing
from tvm.relay.testing.tf import vmobj_to_list as vmobj_to_list

import tensorflow as tf
from tensorflow.python.eager.def_function import Function


def run_tf_code(func, input_):
    if type(func) is Function:
        f_out = func(input_)
        if isinstance(f_out, (list, tuple)):
            np_out = [x.numpy() for x in f_out]
        else:
            np_out = [f_out.numpy()]
    else:
        f_out = func(tf.constant(input_))
        if type(f_out) is dict:
            np_out = [f_out[k].numpy() for k in sorted(f_out.keys())]
        elif type(f_out) is list:
            np_out = [x.numpy() for x in f_out]
        else:
            np_out = f_out.numpy()
    return np_out


def compile_graph_executor(mod, params, target="llvm", target_host="llvm", opt_level=3):
    with tvm.transform.PassContext(opt_level):
        lib = relay.build(mod, target=tvm.target.Target(target, host=target_host), params=params)
    return lib


def compile_vm(mod, params, target="llvm", target_host="llvm", opt_level=3, disabled_pass=None):
    with tvm.transform.PassContext(opt_level, disabled_pass=disabled_pass):
        vm_exec = relay.vm.compile(
            mod, target=tvm.target.Target(target, host=target_host), params=params
        )
    return vm_exec


def run_vm(vm_exec, input_, ctx=tvm.cpu(0)):
    vm = VirtualMachine(vm_exec, ctx)
    _out = vm.invoke("main", input_)
    return vmobj_to_list(_out)


def run_graph_executor(lib, input_, ctx=tvm.cpu(0)):
    mod = runtime.GraphModule(lib["default"](ctx))
    mod.set_input(0, input_)
    mod.run()
    return [mod.get_output(i).numpy() for i in range(mod.get_num_outputs())]


def compare_tf_tvm(gdef, input_, output_, runtime="vm", output_tensors=None):
    """compare tf and tvm execution for the same input.

    Parameters
    ----------
    gdef: TF2 graph def extracted to be fed into from_tensorflow parser.
    (https://www.tensorflow.org/code/tensorflow/core/framework/graph.proto)

    input_: a single numpy array object

    output_: the expected output from TF to match TVM output with

    runtime: choose TVM runtime; either "vm" for VirtualMachine or "graph" for GraphExecutor

    output_tensors : List of output tensor names (Optional)
            if not specified then the last node is assumed as graph output.
    """
    mod, params = from_tensorflow(gdef, outputs=output_tensors)
    if runtime == "vm":
        exec_ = compile_vm(mod, params)
        tvm_out = run_vm(exec_, input_)
    elif runtime == "graph":
        lib = compile_graph_executor(mod, params)
        tvm_out = run_graph_executor(lib, input_)
    else:
        raise RuntimeError("Runtime input not supported: %s" % runtime)

    tvm.testing.assert_allclose(output_, tvm_out, atol=1e-5)
