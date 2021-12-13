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
# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name, W0611, C0413

"""Expose MRVL test functions to the Python frontend"""

import os
import json
import re

import tvm
from tvm import relay
from tvm.relay.op.contrib import mrvl


def file_exists(full_path_filename):
    """Check existance of given file."""
    return os.path.exists(full_path_filename) and os.path.isfile(full_path_filename)


def get_cpu_op_count(mod):
    """Traverse graph counting ops offloaded to TVM."""

    class Counter(tvm.relay.ExprVisitor):
        def __init__(self):
            super().__init__()
            self.count = 0

        def visit_call(self, call):
            if isinstance(call.op, tvm.ir.Op):
                self.count += 1

            super().visit_call(call)

    c = Counter()
    c.visit(mod["main"])
    return c.count


def skip_json_codegen_test():
    """Skip test if it requires the Mrvl codegen and it's not present."""
    # Mrvl codegen not present.
    if not tvm.get_global_func("relay.ext.mrvl", True):
        print("Skip because Mrvl codegen is not available.")
        return True


def skip_aot_runtime_test():
    """Skip test if it requires the Mrvl runtime and it's not present."""
    # Mrvl codegen not present.
    if skip_json_codegen_test():
        return True
    if not mrvl.is_mrvl_runtime_enabled():
        print("Skip because Mrvl runtime isn't present or a remote device isn't being used.")
        return True
    return False


def aot_build_and_json_codegen(
    model_name,
    working_dir,
    raw_model_ir,
    weight_bias_params,
    defuse_mrvl_layers_list=[],
):
    gen_non_mrvl_subgraph = True
    if defuse_mrvl_layers_list is []:
        gen_non_mrvl_subgraph = False

    # produce at most two subgraphs, one mrvl subgraph and/or one non-mrvl subgraph
    try:
        (
            model_mrvl,
            model_other,
            orig_params,
            opt_level,
            disabled_pass,
            orig_mod,
            mrvl_layers_in_mrvl_subgraph,
        ) = mrvl.partition_for_mrvl(
            raw_model_ir,
            params=weight_bias_params,
            tvm_custom_dict={},
            gen_non_mrvl_subgraph=gen_non_mrvl_subgraph,
            flow_pass=1,
        )
        assert orig_params is not None
        assert opt_level is not None
        assert orig_mod is not None
    except Exception as e:
        err_msg = f"The {model_name} model could not be partitioned into subgraph.\n"
        err_msg += str(e)
        raise Exception(err_msg)

    try:
        build_target, device_id = "llvm", 0
        mod_name = relay.backend.utils.mangle_module_name("")
        byoc_executor = relay.build(model_mrvl, target=build_target, mod_name=mod_name)
        #
        byoc_const_params = byoc_executor.get_params()
        byoc_external_graph_json = byoc_executor.get_external_graph_json()
        assert byoc_const_params is not None
        assert byoc_external_graph_json is not None
    except Exception as e:
        err_msg = f"Subgraph(s) could not be relay.build.\n"
        err_msg += str(e)
        raise Exception(err_msg)

    try:
        nodes_json_filename, consts_json_filename = mrvl.dump_json_meta_data_files(
            byoc_external_graph_json,
            byoc_const_params,
            filename_prefix=f"{working_dir}{model_name}-tvm-mrvl-byoc-ir",
        )
        assert nodes_json_filename
        assert consts_json_filename
    except Exception as e:
        err_msg = f"Mrvl JSON codegen failed.\n"
        err_msg += str(e)
        raise Exception(err_msg)

    return (
        nodes_json_filename,
        consts_json_filename,
        model_mrvl,
        model_other,
        mrvl_layers_in_mrvl_subgraph,
        # FIXME: to return mrvl_layers_in_non_mrvl_subgraph
        [],
    )


def check_json_integrity(nodes_json_file):
    json_obj = json.load(open(nodes_json_file, "r"))
    assert "nodes" in json_obj, f"No nodes_json_file['nodes']"
    assert "heads" in json_obj, f"No nodes_json_file['heads']"

    legal_layer_name_list = [
        "input",
        "layout_transform",
        "nn.batch_flatten",
        "reshape",
        "transpose",
        "Avgpool2D",
        "Conv2D",
        "FC",
        "Maxpool2D",
        "Sum2D",
    ]
    mod_name = relay.backend.utils.mangle_module_name("")
    name_regex = "(?P<name>" + mod_name + "_mrvl_main_[0-9]+)"
    for layer in json_obj["nodes"]:
        assert "attrs" in layer, f"No json_obj['attrs']"
        assert "layer_name" in layer["attrs"], f"No json_obj['attrs']['layer_name']"
        layer_name = layer["attrs"]["layer_name"][0]
        assert layer_name in legal_layer_name_list, f"Illegal layer name {layer_name}"
        #
        if layer_name != "input":
            assert "name" in layer, f"No json_obj['name']"
            assert re.match(name_regex, layer["name"]), f"Illegal name ({layer['name']})"

    return json_obj


def verify_json_codegen(nodes_json_file, model_verification_info={}):
    """verify json codegen output JSON files."""
    assert nodes_json_file != ""
    assert file_exists(nodes_json_file), f"{nodes_json_file} does not exist"
    #
    json_obj = check_json_integrity(nodes_json_file)
    if model_verification_info is not {}:
        if "heads_size" in model_verification_info:
            expected_size = model_verification_info["heads_size"]
            actual_size = len(json_obj["heads"])
            assert (
                actual_size == expected_size
            ), f"heads size - expected {expected_size} != actual {actual_size}"
        if "nodes_size" in model_verification_info:
            expected_size = model_verification_info["nodes_size"]
            actual_size = len(json_obj["nodes"])
            assert (
                actual_size == expected_size
            ), f"nodes size - expected {expected_size} != actual {actual_size}"


def aot_runtime_gen(
    nodes_json_filename,
    consts_json_filename,
    aot_fp16_cmd_opts,
):
    """aot runtime gen."""
    # TODO(ccjoechou): add final code
    mrvl_subgraph_runtime_model_binary = None
    assert nodes_json_filename is not None
    assert consts_json_filename is not None
    assert aot_fp16_cmd_opts is not None
    return mrvl_subgraph_runtime_model_binary


def aot_run(mrvl_subgraph_runtime_model_binary, aot_run_cmd_opts, inf_inp=[]):
    """mrvl aot run output."""
    # TODO(ccjoechou): add final code
    mrvl_subgraph_actual_fp16_output = []
    assert mrvl_subgraph_runtime_model_binary is not None
    assert aot_run_cmd_opts is not None
    assert inf_inp is not []
    return mrvl_subgraph_actual_fp16_output


def tvm_llvm_fp32_run(mod_mrvl_subgraph, mrvl_layers_in_mrvl_subgraph, inf_inp=[]):
    """llvm run output."""
    # TODO(ccjoechou): add final code
    mrvl_subgraph_golden_fp32_output = []
    assert mod_mrvl_subgraph is not None
    assert mrvl_layers_in_mrvl_subgraph is not None
    assert data_inp is not []
    return mrvl_subgraph_golden_fp32_output


def verify_mrvl_subgraph_aot_inf_result(
    mrvl_subgraph_actual_fp16_output,
    mrvl_subgraph_golden_fp32_output,
    delta_config,
):
    """verify inf output of mrvl subgraph."""
    # TODO(ccjoechou): add final code
    assert mrvl_subgraph_actual_fp16_output is not None
    assert mrvl_subgraph_golden_fp32_output is not None
    assert delta_config is not None
    return


def verify_aot_inf_result(actual_inf_output, delta_config):
    """verify final inf output of model."""
    # TODO(ccjoechou): add final code
    assert actual_inf_output is not None
    assert delta_config is not None
