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
"""NNAPI ComputeDevice specialization
"""
import numpy as np
import tvm
from tvm.contrib.target.android_nnapi.relayir_to_nnapi_converter import convert_relayir_to_nnapi
from tvm.contrib.target.android_nnapi.relayir_to_nnapi_converter.error import (
    AndroidNNAPICompilerIncompatibleError,
)
from ....._base import post_partition_transform
from ._rpc_device import RPCDevice
from ._error import AndroidNNAPICompilerProfilingError
from . import _utils


def _isolate_op_call_node(call, compiler):
    func_params = []
    new_call_args = []
    for i, arg in enumerate(call.args):
        if isinstance(arg.checked_type, tvm.relay.TupleType):
            tuple_param_fields = [
                tvm.relay.var(f"arg{ i }.{ j }", type_annotation=f)
                for j, f in enumerate(arg.checked_type.fields)
            ]
            func_params += tuple_param_fields
            tuple_arg = tvm.relay.Tuple(
                [tvm.relay.annotation.compiler_begin(f, compiler) for f in tuple_param_fields]
            )
            new_call_args.append(tuple_arg)
        elif isinstance(arg.checked_type, tvm.relay.TensorType):
            func_params.append(tvm.relay.var(f"arg{ i }", type_annotation=arg.checked_type))
            new_call_args.append(tvm.relay.annotation.compiler_begin(func_params[-1], compiler))
        else:
            raise NotImplementedError(arg.checked_type)
    new_call = tvm.relay.annotation.compiler_end(
        tvm.relay.Call(call.op, new_call_args, call.attrs, call.type_args), compiler
    )
    return tvm.relay.Function(func_params, new_call)


class NnapiDevice(RPCDevice):
    """NNAPI ComputeDevice specialization"""

    DEV_NAME = "nnapi"

    def __init__(self, options, tracker):
        super().__init__(options, tracker)
        self._api_level = options["target"]["api_level"]
        self._compiler_name = options["tvm"]["external_compiler"]

    def estimate_call_op_cost(self, call):
        assert isinstance(call.op, tvm.ir.Op)

        # prepare the module to run
        mod = tvm.IRModule({"main": _isolate_op_call_node(call, self._compiler_name)})
        mod = tvm.relay.transform.PartitionGraph()(mod)

        # get runtime on device (or failure)
        try:
            return self._get_runtime_on_device(mod)
        except AndroidNNAPICompilerProfilingError:
            return None

    def estimate_single_byte_read_cost_to_bus(self):
        return self._data_transfer_to_main_memory_cost

    def estimate_single_byte_write_cost_to_bus(self):
        return self._data_transfer_to_main_memory_cost

    def _get_runtime_on_device(self, mod):
        assert isinstance(mod, tvm.IRModule)

        mod = tvm.relay.transform.InferType()(mod)
        if isinstance(mod["main"].ret_type, tvm.relay.TensorType):
            # prepare params
            params = {
                p.name_hint: tvm.nd.array(
                    np.random.uniform(size=tuple([int(i) for i in p.checked_type.shape])).astype(
                        str(p.checked_type.dtype)
                    ),
                    tvm.cpu(0),
                )
                for p in mod["main"].params
            }

            # run some post partition transformation and fixes
            # here we try to mimic the result of an partition
            mod, params = post_partition_transform(
                mod,
                params,
                android_nnapi_level=self._options["target"]["api_level"],
                external_compiler=self._options["tvm"]["external_compiler"],
            )

            external_func = (lambda op: op if isinstance(op, tvm.relay.Function) else mod[op])(
                mod["main"].body.op
            )  # op may be a GlobalVar, hence the if
            assert isinstance(external_func, tvm.relay.Function)
            external_func = external_func.with_attr(
                "NnapiClassName", f"{ external_func.attrs.global_symbol }_0"
            )  # NnapiClassName is required for the converter

            # try converting first to see if there's any problem
            # if there's any incompatible case, an error would be thrown
            try:
                convert_relayir_to_nnapi(external_func)
            except AndroidNNAPICompilerIncompatibleError as err:
                raise AndroidNNAPICompilerProfilingError(
                    f"Relay operator unsupported by Android NNAPI converter: { str(err) }"
                )

            # build binary
            mod = tvm.relay.transform.InferType()(mod)
            with tvm.transform.PassContext(opt_level=3):
                exe = tvm.relay.vm.compile(mod, target=self._tvm_target)
            _, lib = exe.save()
            assert lib

            temp_dir = tvm.contrib.utils.tempdir()
            temp_lib_path = temp_dir.relpath("lib.so")

            def _scope():
                kwargs = {}
                kwargs["options"] = [
                    "--target={}".format(self._target_triple),
                    "-O3",
                    "-lneuralnetworks",
                    "-shared",
                    "-fPIC",
                ]
                lib.export_library(temp_lib_path, fcompile=tvm.contrib.ndk.create_shared, **kwargs)

            _scope()

            # push binary
            remote = self._tracker.request(self._remote_key)
            remote.upload(temp_lib_path)
            remote_mod = remote.load_module("lib.so")

            # run
            device = remote.cpu()
            args = [params[p.name_hint] for p in mod["main"].params]
            args.append(
                _utils.get_function_output_buffer(external_func, device)
            )  # arg contains an additional output buffer at the end
            remote_func = remote_mod.time_evaluator(
                str(external_func.attrs.global_symbol), device, number=self._remote_profile_run
            )
            ret = remote_func(*args).mean
        elif isinstance(mod["main"].ret_type, tvm.relay.TupleType):
            # Tuple(ADT) is not supported by RPC (and NNAPI!)
            raise AndroidNNAPICompilerProfilingError(f"Relay tuple-typed operator is unsupported")
        else:
            raise NotImplementedError(str(mod["main"].ret_type))

        return ret

    @property
    def _data_transfer_to_main_memory_cost(self):  # pylint: disable=invalid-name
        if getattr(self, "_data_transfer_to_main_memory_cost_val", None) is not None:
            return (
                self._data_transfer_to_main_memory_cost_val  # pylint: disable=access-member-before-definition
            )
        # lazy init
        comm_node_size = [0]
        time_statistics = {}
        # benchmark for a single conv_2d (|-|)
        def _scope():
            img = tvm.relay.var("img", shape=[32, 512, 512, 1], dtype="float32")
            ann_img = tvm.relay.annotation.compiler_begin(img, self._compiler_name)
            weight_0 = tvm.relay.var("weight_0", shape=[1, 1, 1, 1], dtype="float32")
            ann_weight_0 = tvm.relay.annotation.compiler_begin(weight_0, self._compiler_name)
            conv_0 = tvm.relay.nn.conv2d(
                ann_img, ann_weight_0, data_layout="NHWC", kernel_layout="OHWI"
            )
            ann_conv_0 = tvm.relay.annotation.compiler_end(conv_0, self._compiler_name)
            single_conv_f = tvm.relay.Function([img, weight_0], ann_conv_0)
            mod = tvm.IRModule({"main": single_conv_f})
            mod = tvm.relay.transform.PartitionGraph()(mod)

            # get comm_node_size
            mod = tvm.relay.transform.InferType()(mod)
            comm_node_size[0] = _utils.get_node_size(mod["main"].body)

            time_statistics["single_conv"] = self._get_runtime_on_device(mod)

        _scope()

        def _scope():  # benchmark for 2 conv_2ds (|--|)
            img = tvm.relay.var("img", shape=[32, 512, 512, 1], dtype="float32")
            ann_img = tvm.relay.annotation.compiler_begin(img, self._compiler_name)
            weight_0 = tvm.relay.var("weight_0", shape=[1, 1, 1, 1], dtype="float32")
            ann_weight_0 = tvm.relay.annotation.compiler_begin(weight_0, self._compiler_name)
            conv_0 = tvm.relay.nn.conv2d(
                ann_img, ann_weight_0, data_layout="NHWC", kernel_layout="OHWI"
            )
            weight_1 = tvm.relay.var("weight_1", shape=[1, 1, 1, 1], dtype="float32")
            ann_weight_1 = tvm.relay.annotation.compiler_begin(weight_1, self._compiler_name)
            conv_1 = tvm.relay.nn.conv2d(
                conv_0, ann_weight_1, data_layout="NHWC", kernel_layout="OHWI"
            )
            ann_conv_1 = tvm.relay.annotation.compiler_end(conv_1, self._compiler_name)
            two_conv_f = tvm.relay.Function([img, weight_0, weight_1], ann_conv_1)
            mod = tvm.IRModule({"main": two_conv_f})
            mod = tvm.relay.transform.PartitionGraph()(mod)
            time_statistics["two_conv"] = self._get_runtime_on_device(mod)

        _scope()

        self._data_transfer_to_main_memory_cost_val = (  # pylint: disable=invalid-name
            time_statistics["single_conv"] - time_statistics["two_conv"] / 2
        ) / comm_node_size[
            0
        ]  # diff(|-||-|, |--|) / 2 / size-of-tensor
        return self._data_transfer_to_main_memory_cost_val
