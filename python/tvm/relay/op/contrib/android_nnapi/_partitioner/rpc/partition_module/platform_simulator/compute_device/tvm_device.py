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
"""TVM ComputeDevice specialization."""
import numpy as np
import tvm
from ._rpc_device import RPCDevice
from . import _utils


def _isolate_op_call_node(call):
    func_params = []
    new_call_args = []
    for i, arg in enumerate(call.args):
        if isinstance(arg.checked_type, tvm.relay.TupleType):
            tuple_param_fields = [
                tvm.relay.var(f"arg{ i }.{ j }", type_annotation=f)
                for j, f in enumerate(arg.checked_type.fields)
            ]
            func_params += tuple_param_fields
            tuple_arg = tvm.relay.Tuple(tuple_param_fields)
            new_call_args.append(tuple_arg)
        elif isinstance(arg.checked_type, tvm.relay.TensorType):
            func_params.append(tvm.relay.var(f"arg{ i }", type_annotation=arg.checked_type))
            new_call_args.append(func_params[-1])
        else:
            raise NotImplementedError(arg.checked_type)
    new_call = tvm.relay.Call(call.op, new_call_args, call.attrs, call.type_args)
    return tvm.relay.Function(func_params, new_call)


class TvmDevice(RPCDevice):
    """TVM ComputeDevice specialization."""

    DEV_NAME = "tvm"

    def estimate_call_op_cost(self, call):
        assert isinstance(call.op, tvm.ir.Op)

        mod = tvm.IRModule({"main": _isolate_op_call_node(call)})
        mod = tvm.relay.transform.InferType()(mod)

        return self._get_runtime_on_device(mod)

    def _get_runtime_on_device(self, mod):
        assert isinstance(mod, tvm.IRModule)

        mod = tvm.relay.transform.InferType()(mod)
        if isinstance(mod["main"].ret_type, tvm.relay.TensorType):
            with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
                exe = tvm.relay.vm.compile(mod, target=self._tvm_target)
            _, lib = exe.save()

            if not lib:
                return 0

            temp_dir = tvm.contrib.utils.tempdir()
            temp_lib_path = temp_dir.relpath("lib.so")

            def _scope():
                kwargs = {}
                kwargs["options"] = [
                    "--target={}".format(self._target_triple),
                    "-O3",
                    "-shared",
                    "-fPIC",
                ]
                lib.export_library(temp_lib_path, fcompile=tvm.contrib.ndk.create_shared, **kwargs)

            _scope()

            remote = self._tracker.request(self._remote_key)
            remote.upload(temp_lib_path)
            remote_mod = remote.load_module("lib.so")

            device = remote.cpu()
            args = [
                tvm.nd.array(
                    np.random.uniform(size=tuple([int(i) for i in p.checked_type.shape])).astype(
                        str(p.checked_type.dtype)
                    ),
                    device,
                )
                for p in mod["main"].params
            ]
            args.append(_utils.get_function_output_buffer(mod["main"], device))  # output buffer

            def _scope():
                primitives = exe.primitive_ops
                assert len(primitives) == 1
                return primitives[0]

            main_sym = _scope()
            remote_func = remote_mod.time_evaluator(
                main_sym, device, number=self._remote_profile_run
            )
            ret = remote_func(*args).mean
        elif isinstance(mod["main"].ret_type, tvm.relay.TupleType):
            # Tuple(ADT) is not supported by RPC
            ret = 0
        else:
            raise NotImplementedError(mod["main"].ret_type)
        return ret

    def estimate_memory_read_cost(self, dtype, size):
        return 0

    def estimate_memory_write_cost(self, dtype, size):
        return 0
