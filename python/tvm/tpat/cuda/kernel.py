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

import tvm
import tvm.contrib.graph_executor as runtime
import tvm.relay as relay
from tvm import dlight
from tvm import meta_schedule as ms


class Config(object):
    def __init__(self, onnx_model, input_shapes, target, work_dir) -> None:
        self.onnx_model = onnx_model
        self.input_shapes = input_shapes
        self.work_dir = work_dir

        if target == "gpu":
            self.target = self._detect_cuda_target()

    def tune_option(self):
        return {
            "target": self.target,
            "builder": ms.builder.LocalBuilder(),
            "runner": ms.runner.LocalRunner(),
            "max_trials_global": 1000,
            "max_trials_per_task": 100,
            "work_dir": self.work_dir,
        }

    def _detect_cuda_target(self):
        dev = tvm.cuda()
        if not dev.exist:
            return None

        return tvm.target.Target(
            {
                "kind": "cuda",
                "max_shared_memory_per_block": dev.max_shared_memory_per_block,
                "max_threads_per_block": dev.max_threads_per_block,
                "thread_warp_size": dev.warp_size,
                "registers_per_block": 65536,
                "arch": "sm_" + tvm.cuda().compute_version.replace(".", ""),
            }
        )


class Kernel(object):
    def __init__(self, name, onnx_model, input_shapes, enable_tunning, work_dir):
        self._name = name
        self._enable_tunning = enable_tunning
        self._config = Config(onnx_model, input_shapes, "gpu", work_dir)

        self._lib = None
        self._module = None

    def run(self):
        """
        Tvm Auto Scheduler
        """

        # 1. Model -> Relay
        mod, params = relay.frontend.from_onnx(self._config.onnx_model)

        # 2. Tune it
        if self._enable_tunning:
            tunning_option = self._config.tune_option()
            ms.relay_integration.tune_relay(mod=mod, params=params, **tunning_option)

        # 3. Compiling
        try:
            if self._enable_tunning:
                db = ms.Database.create(kind="json", work_dir=self._config.work_dir)
                with db, self._config.target as target, tvm.transform.PassContext(opt_level=3):
                    mod = dlight.ApplyDefaultSchedule(dlight.gpu.Fallback())(mod)  # type: ignore
                    mod = tvm.tir.transform.ForceNarrowIndexToInt32()(mod)
                    lib = ms.relay_integration.compile_relay(
                        database=db,
                        mod=mod,
                        target=target,
                        params=params,
                    )
            else:
                with self._config.target as target, tvm.transform.PassContext(opt_level=3):
                    mod = dlight.ApplyDefaultSchedule(dlight.gpu.Fallback())(mod)  # type: ignore
                    mod = tvm.tir.transform.ForceNarrowIndexToInt32()(mod)
                    lib = relay.build(mod, target=target, params=params)

            # load parameters
            dev = tvm.cuda(0)
            module_exec = runtime.GraphModule(lib["default"](dev))  # type: ignore

            self._lib = lib
            self._module = module_exec

            # 4. Running
            self._module.run()
        except Exception as e:
            print("[ERROR]: ", e)
            self._lib = None
            self._module = None

    @property
    def cuda_source_code(self):
        """Return source code of this kernel.

        Returns
        -------
        str
            source code of kernel
        """
        if not self._lib:
            return None

        try:
            source_code = self._lib.get_lib().imported_modules[0].get_source()
            source_code = source_code.replace("signed char*", "int*")
            source_code = source_code.replace("uint64_t*", "int*")
            source_code = source_code.replace("long long", "int")
            source_code = source_code.replace("double", "float")
        except IndexError:
            return None
        return source_code

    @property
    def runtime_module(self):
        return self._lib

    @property
    def graph_module(self):
        return self._module

    @property
    def constant_param(self):
        return self._lib.get_constant_params() if self._lib else None

    @property
    def device_funcs_inorder(self):
        return self._lib.get_device_function_list() if self._lib else None

    @property
    def device_funcs_thread_config(self):
        return self._lib.get_grid_block_thread_config() if self._lib else None

    @property
    def device_allocate_global_memory(self):
        return self._lib.get_device_memory_size() if self._lib else None

    @property
    def num_inputs(self):
        return self._module.get_num_inputs() if self._module else None

    @property
    def num_outputs(self):
        return self._module.get_num_outputs() if self._module else None

    @property
    def workspace_dtype(self):
        return self._module.get_workspace_dtype() if self._module else None

    @property
    def workspace_size(self):
        return self._module.get_workspace_size() if self._module else None

    @property
    def func_inorder(self):
        return self._module.get_func_inorder() if self._module else None

    @property
    def storageid(self):
        return self._module.get_storageid() if self._module else None

    @property
    def plugin_name(self):
        return self._name
