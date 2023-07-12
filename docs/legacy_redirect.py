# -*- coding: utf-8 -*-

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

from string import Template
import json
import os

legacy_redirects = [
    ["dev/benchmark.html", "../arch/benchmark.html"],
    ["dev/convert_layout.html", "../arch/convert_layout.html"],
    ["dev/debugger.html", "../arch/debugger.html"],
    ["dev/device_target_interactions.html", "../arch/device_target_interactions.html"],
    ["dev/frontend/tensorflow.html", "../../arch/frontend/tensorflow.html"],
    ["dev/hybrid_script.html", "../arch/hybrid_script.html"],
    ["dev/index.html", "../arch/index.html"],
    ["dev/inferbound.html", "../arch/inferbound.html"],
    [
        "dev/introduction_to_module_serialization.html",
        "../arch/introduction_to_module_serialization.html",
    ],
    ["dev/microtvm_design.html", "../arch/microtvm_design.html"],
    ["dev/model_library_format.html", "../arch/model_library_format.html"],
    ["dev/pass_infra.html", "../arch/pass_infra.html"],
    ["dev/relay_intro.html", "../arch/relay_intro.html"],
    ["dev/relay_op_strategy.html", "../arch/relay_op_strategy.html"],
    ["dev/runtime.html", "../arch/runtime.html"],
    ["dev/runtimes/vulkan.html", "../../arch/runtimes/vulkan.html"],
    ["dev/security.html", "../arch/security.html"],
    ["dev/virtual_machine.html", "../arch/virtual_machine.html"],
    ["dev/how_to.html", "index.html"],
    ["dev/pytest_target_parametrization.html", "how_to/pytest_target_parametrization.html"],
    ["dev/relay_add_op.html", "how_to/relay_add_op.html"],
    ["dev/relay_add_pass.html", "how_to/relay_add_pass.html"],
    ["dev/relay_bring_your_own_codegen.html", "how_to/relay_bring_your_own_codegen.html"],
    ["dev/codebase_walkthrough.html", "tutorial/codebase_walkthrough.html"],
    ["deploy/android.html", "../how_to/deploy/android.html"],
    ["deploy/arm_compute_lib.html", "../how_to/deploy/arm_compute_lib.html"],
    ["deploy/bnns.html", "../how_to/deploy/bnns.html"],
    ["deploy/cpp_deploy.html", "../how_to/deploy/cpp_deploy.html"],
    ["deploy/hls.html", "../how_to/deploy/hls.html"],
    ["deploy/index.html", "../how_to/deploy/index.html"],
    ["deploy/integrate.html", "../how_to/deploy/integrate.html"],
    ["deploy/tensorrt.html", "../how_to/deploy/tensorrt.html"],
    ["deploy/vitis_ai.html", "../how_to/deploy/vitis_ai.html"],
    ["profiling/index.html", "../how_to/profile/index.html"],
    ["profiling/papi.html", "../how_to/profile/papi.html"],
    ["api/links.html", "../reference/api/links.html"],
    ["api/python/auto_scheduler.html", "../../reference/api/python/auto_scheduler.html"],
    ["api/python/autotvm.html", "../../reference/api/python/autotvm.html"],
    ["api/python/contrib.html", "../../reference/api/python/contrib.html"],
    ["api/python/driver.html", "../../reference/api/python/driver.html"],
    ["api/python/error.html", "../../reference/api/python/error.html"],
    ["api/python/graph_executor.html", "../../reference/api/python/graph_executor.html"],
    ["api/python/index.html", "../../reference/api/python/index.html"],
    ["api/python/ir.html", "../../reference/api/python/ir.html"],
    ["api/python/micro.html", "../../reference/api/python/micro.html"],
    ["api/python/ndarray.html", "../../reference/api/python/ndarray.html"],
    ["api/python/relay/analysis.html", "../../../reference/api/python/relay/analysis.html"],
    ["api/python/relay/backend.html", "../../../reference/api/python/relay/backend.html"],
    [
        "api/python/relay/dataflow_pattern.html",
        "../../../reference/api/python/relay/dataflow_pattern.html",
    ],
    ["api/python/relay/frontend.html", "../../../reference/api/python/relay/frontend.html"],
    ["api/python/relay/image.html", "../../../reference/api/python/relay/image.html"],
    ["api/python/relay/index.html", "../../../reference/api/python/relay/index.html"],
    ["api/python/relay/nn.html", "../../../reference/api/python/relay/nn.html"],
    ["api/python/relay/testing.html", "../../../reference/api/python/relay/testing.html"],
    ["api/python/relay/transform.html", "../../../reference/api/python/relay/transform.html"],
    ["api/python/relay/vision.html", "../../../reference/api/python/relay/vision.html"],
    ["api/python/rpc.html", "../../reference/api/python/rpc.html"],
    ["api/python/runtime.html", "../../reference/api/python/runtime.html"],
    ["api/python/target.html", "../../reference/api/python/target.html"],
    ["api/python/te.html", "../../reference/api/python/te.html"],
    ["api/python/tir.html", "../../reference/api/python/tir.html"],
    ["api/python/topi.html", "../../reference/api/python/topi.html"],
    ["api/python/vta/index.html", "../../../reference/api/python/vta/index.html"],
    ["langref/hybrid_script.html", "../reference/langref/hybrid_script.html"],
    ["langref/index.html", "../reference/langref/index.html"],
    ["langref/relay_adt.html", "../reference/langref/relay_adt.html"],
    ["langref/relay_expr.html", "../reference/langref/relay_expr.html"],
    ["langref/relay_op.html", "../reference/langref/relay_op.html"],
    ["langref/relay_pattern.html", "../reference/langref/relay_pattern.html"],
    ["langref/relay_type.html", "../reference/langref/relay_type.html"],
    ["microtvm/index.html", "../topic/microtvm/index.html"],
    ["vta/dev/config.html", "../../topic/vta/dev/config.html"],
    ["vta/dev/hardware.html", "../../topic/vta/dev/hardware.html"],
    ["vta/dev/index.html", "../../topic/vta/dev/index.html"],
    ["vta/index.html", "../topic/vta/index.html"],
    ["vta/install.html", "../topic/vta/install.html"],
    ["tutorials/index.html", "../tutorial/index.html"],
    ["tutorials/frontend/from_caffe2.html", "../../how_to/compile_models/from_caffe2.html"],
    ["tutorials/frontend/from_coreml.html", "../../how_to/compile_models/from_coreml.html"],
    ["tutorials/frontend/from_darknet.html", "../../how_to/compile_models/from_darknet.html"],
    ["tutorials/frontend/from_keras.html", "../../how_to/compile_models/from_keras.html"],
    ["tutorials/frontend/from_mxnet.html", "../../how_to/compile_models/from_mxnet.html"],
    ["tutorials/frontend/from_onnx.html", "../../how_to/compile_models/from_onnx.html"],
    ["tutorials/frontend/from_paddle.html", "../../how_to/compile_models/from_paddle.html"],
    ["tutorials/frontend/from_pytorch.html", "../../how_to/compile_models/from_pytorch.html"],
    ["tutorials/frontend/from_tensorflow.html", "../../how_to/compile_models/from_tensorflow.html"],
    ["tutorials/frontend/from_tflite.html", "../../how_to/compile_models/from_tflite.html"],
    [
        "tutorials/frontend/deploy_model_on_android.html",
        "../../how_to/deploy_models/deploy_model_on_android.html",
    ],
    [
        "tutorials/frontend/deploy_model_on_rasp.html",
        "../../how_to/deploy_models/deploy_model_on_rasp.html",
    ],
    [
        "tutorials/frontend/deploy_object_detection_pytorch.html",
        "../../how_to/deploy_models/deploy_object_detection_pytorch.html",
    ],
    [
        "tutorials/frontend/deploy_prequantized.html",
        "../../how_to/deploy_models/deploy_prequantized.html",
    ],
    [
        "tutorials/frontend/deploy_prequantized_tflite.html",
        "../../how_to/deploy_models/deploy_prequantized_tflite.html",
    ],
    [
        "tutorials/frontend/deploy_quantized.html",
        "../../how_to/deploy_models/deploy_quantized.html",
    ],
    ["tutorials/frontend/deploy_sparse.html", "../../how_to/deploy_models/deploy_sparse.html"],
    [
        "tutorials/dev/bring_your_own_datatypes.html",
        "../../how_to/extend_tvm/bring_your_own_datatypes.html",
    ],
    [
        "tutorials/dev/low_level_custom_pass.html",
        "../../how_to/extend_tvm/low_level_custom_pass.html",
    ],
    ["tutorials/dev/use_pass_infra.html", "../../how_to/extend_tvm/use_pass_infra.html"],
    ["tutorials/dev/use_pass_instrument.html", "../../how_to/extend_tvm/use_pass_instrument.html"],
    ["tutorials/optimize/opt_conv_cuda.html", "../../how_to/optimize_operators/opt_conv_cuda.html"],
    [
        "tutorials/optimize/opt_conv_tensorcore.html",
        "../../how_to/optimize_operators/opt_conv_tensorcore.html",
    ],
    ["tutorials/optimize/opt_gemm.html", "../../how_to/optimize_operators/opt_gemm.html"],
    [
        "tutorials/auto_scheduler/tune_conv2d_layer_cuda.html",
        "../../how_to/tune_with_autoscheduler/tune_conv2d_layer_cuda.html",
    ],
    [
        "tutorials/auto_scheduler/tune_network_arm.html",
        "../../how_to/tune_with_autoscheduler/tune_network_arm.html",
    ],
    [
        "tutorials/auto_scheduler/tune_network_cuda.html",
        "../../how_to/tune_with_autoscheduler/tune_network_cuda.html",
    ],
    [
        "tutorials/auto_scheduler/tune_network_mali.html",
        "../../how_to/tune_with_autoscheduler/tune_network_mali.html",
    ],
    [
        "tutorials/auto_scheduler/tune_network_x86.html",
        "../../how_to/tune_with_autoscheduler/tune_network_x86.html",
    ],
    [
        "tutorials/auto_scheduler/tune_sparse_x86.html",
        "../../how_to/tune_with_autoscheduler/tune_sparse_x86.html",
    ],
    [
        "tutorials/autotvm/tune_conv2d_cuda.html",
        "../../how_to/tune_with_autotvm/tune_conv2d_cuda.html",
    ],
    ["tutorials/autotvm/tune_relay_arm.html", "../../how_to/tune_with_autotvm/tune_relay_arm.html"],
    [
        "tutorials/autotvm/tune_relay_cuda.html",
        "../../how_to/tune_with_autotvm/tune_relay_cuda.html",
    ],
    [
        "tutorials/autotvm/tune_relay_mobile_gpu.html",
        "../../how_to/tune_with_autotvm/tune_relay_mobile_gpu.html",
    ],
    ["tutorials/autotvm/tune_relay_x86.html", "../../how_to/tune_with_autotvm/tune_relay_x86.html"],
    ["tutorials/micro/micro_autotune.html", "../../how_to/work_with_microtvm/micro_autotune.html"],
    [
        "tutorials/micro/micro_reference_vm.html",
        "../../how_to/work_with_microtvm/micro_reference_vm.html",
    ],
    ["tutorials/micro/micro_tflite.html", "../../how_to/work_with_microtvm/micro_tflite.html"],
    ["tutorials/frontend/build_gcn.html", "../../how_to/work_with_relay/build_gcn.html"],
    [
        "tutorials/frontend/using_external_lib.html",
        "../../how_to/work_with_relay/using_external_lib.html",
    ],
    ["tutorials/language/extern_op.html", "../../how_to/work_with_schedules/extern_op.html"],
    ["tutorials/language/intrin_math.html", "../../how_to/work_with_schedules/intrin_math.html"],
    ["tutorials/language/reduction.html", "../../how_to/work_with_schedules/reduction.html"],
    ["tutorials/language/scan.html", "../../how_to/work_with_schedules/scan.html"],
    [
        "tutorials/language/schedule_primitives.html",
        "../../how_to/work_with_schedules/schedule_primitives.html",
    ],
    ["tutorials/language/tedd.html", "../../how_to/work_with_schedules/tedd.html"],
    ["tutorials/language/tensorize.html", "../../how_to/work_with_schedules/tensorize.html"],
    ["tutorials/language/tuple_inputs.html", "../../how_to/work_with_schedules/tuple_inputs.html"],
    [
        "tutorials/get_started/auto_scheduler_matmul_x86.html",
        "../../tutorial/auto_scheduler_matmul_x86.html",
    ],
    ["tutorials/get_started/autotvm_matmul_x86.html", "../../tutorial/autotvm_matmul_x86.html"],
    ["tutorials/get_started/autotvm_relay_x86.html", "../../tutorial/autotvm_relay_x86.html"],
    [
        "tutorials/get_started/cross_compilation_and_rpc.html",
        "../../tutorial/cross_compilation_and_rpc.html",
    ],
    ["tutorials/get_started/install.html", "../../tutorial/install.html"],
    ["tutorials/topi/intro_topi.html", "../../tutorial/intro_topi.html"],
    ["tutorials/get_started/introduction.html", "../../tutorial/introduction.html"],
    ["tutorials/get_started/relay_quick_start.html", "../../tutorial/relay_quick_start.html"],
    [
        "tutorials/get_started/tensor_expr_get_started.html",
        "../../tutorial/tensor_expr_get_started.html",
    ],
    [
        "tutorials/get_started/tvmc_command_line_driver.html",
        "../../tutorial/tvmc_command_line_driver.html",
    ],
    [
        "tutorials/get_started/tvmc_python.html",
        "../../tutorial/tvmc_python.html",
    ],
]

redirect_template = """
<!DOCTYPE html>
<html>
  <head>
    <meta http-equiv="refresh" content="1; url=$to" />
    <script>
      window.location.href = "$to"
    </script>
  </head>
</html>
"""


def build_legacy_redirect(tvm_path):
    def legacy_redirect(app, docname):  # Sphinx expects two arguments
        if app.builder.name == "html":

            src = Template(redirect_template)

            for frm, to in legacy_redirects:
                frm = tvm_path.resolve() / "docs" / "_build" / "html" / frm
                redirect = src.substitute({"to": to})
                os.makedirs(os.path.dirname(frm), exist_ok=True)
                with open(frm, "w") as f:
                    f.write(redirect)

    return legacy_redirect
