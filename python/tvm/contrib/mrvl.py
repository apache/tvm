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
# pylint: disable=invalid-name, unused-argument, broad-except
"""Utility to compile Marvell models"""

import os
import json
import shutil
import tempfile
import base64
import numpy as np
import tvm
import tvm._ffi


@tvm._ffi.register_func("tvm.mrvl.find_value_in_KV_pair")
def find_value_in_KV_pair(json_input: str, key_to_find: str) -> str:
    """This function takes the graph_json string and key to be searched in
    the json string, using json parser routine it loads the json string
    and access the value using the given key. It raises exception if the
    key is not found in the input json string.

    Parameters
    ----------
    graph_json: String
       This is the graph_json string

    Returns
    -------
    value_string: string
       This returns the value string for the given key string
    """
    value = ""
    try:
        json_dict = json.loads(json_input)
        value = json_dict[key_to_find]
    except KeyError:
        assert False, "Marvell-Compiler-ERROR-Internal:: Could not find matching key in json"

    return value


@tvm._ffi.register_func("tvm.mrvl.GetNodesJSONString")
def get_nodes_json_string(graph_json):
    """This takes the graph_json string from MrvlJSONSerializer and adds / modifies
    the json string to a form suitable for the Marvell Backend.

    Parameters
    ----------
    graph_json: String
        This is the graph_json string from the MrvlJSONSerializer

    Returns
    -------
    nodes_json_string: string
        This returns the nodes_json string which can be accepted by the Marvell backend.
    """

    dictionary = json.loads(graph_json)
    # Add Marvell Index and rename "op" and "name" fields
    mrvl_idx = 1
    num_in = 0
    for iterator in dictionary["nodes"]:
        if iterator["op"] == "kernel":
            iterator["op"] = "tvm_op"
            iterator["attrs"]["mrvl_nodes_idx"] = [mrvl_idx]
            iterator["attrs"]["kernel_const"] = {}
            iterator["attrs"]["bias_const"] = {}
            iterator["attrs"]["beta_const"] = {}
            iterator["attrs"]["gamma_const"] = {}
            iterator["attrs"]["var_const"] = {}
            iterator["attrs"]["mean_const"] = {}
            iterator["name"] = "tvmgen_mrvl_main" + "_" + str(mrvl_idx - 1)
            mrvl_idx = mrvl_idx + 1
        if iterator["op"] == "input":
            iterator["attrs"]["layer_name"] = ["input"]
            iterator["inputs"] = []
            in_id = iterator["name"].split("_i")[-1]
            iterator["input_id"] = [in_id]
            iterator["attrs"]["dtype"] = iterator["attrs"]["dtype"][0]
            iterator["attrs"]["shape"] = iterator["attrs"]["shape"][0]
            if len(iterator["attrs"]["shape"][0]) == 2:
                iterator["attrs"]["data_layout"] = ["NC"]
            else:
                iterator["attrs"]["data_layout"] = ["NCHW"]
            # Infer Batch Size from the input shape
            batch_size = iterator["attrs"]["shape"][0][0]
            dictionary["batch_size"] = f"{batch_size}"
            num_in = num_in + 1

    # Create a new inputs to store only the previous node input and not the const inputs
    for iterator in dictionary["nodes"]:
        if iterator["op"] == "tvm_op":
            list_prev = []
            for prev in iterator["inputs"]:
                if dictionary["nodes"][prev[0]]["op"] == "tvm_op":
                    mrvl_idx_prev = dictionary["nodes"][prev[0]]["attrs"]["mrvl_nodes_idx"][0]
                    list_prev.append([mrvl_idx_prev + num_in - 1, 0, 0])
                if dictionary["nodes"][prev[0]]["op"] == "input":
                    idx_in = int(dictionary["nodes"][prev[0]]["input_id"][0])
                    list_prev.append([idx_in, 0, 0])
            iterator["node_prev"] = list_prev

    for iterator in dictionary["nodes"]:
        if iterator["op"] == "tvm_op":
            del iterator["inputs"]

    for iterator in dictionary["nodes"]:
        if iterator["op"] == "tvm_op":
            iterator["inputs"] = iterator["node_prev"]

    for iterator in dictionary["nodes"]:
        if iterator["op"] == "tvm_op":
            del iterator["node_prev"]

    # Remove unneeded fields
    del dictionary["node_row_ptr"]

    # Patch up arg_nodes and heads to remove references to constant inputs
    list_nodes = dictionary["arg_nodes"]
    list_nodes_updated = []

    for iterator in list_nodes:
        if dictionary["nodes"][iterator]["op"] != "const":
            if dictionary["nodes"][iterator]["op"] == "input":
                input_name = dictionary["nodes"][iterator]["name"]
                input_num_str = input_name.split("_i", 1)[1]
                input_num = int(input_num_str)
                list_nodes_updated.append(input_num)
            else:
                list_nodes_updated.append(
                    dictionary["nodes"][iterator]["attrs"]["mrvl_nodes_idx"][0]
                )
    dictionary["arg_nodes"] = list_nodes_updated

    # Add additional data required by the runtime such as number of inputs
    # and number of outputs to the subgraph
    num_subgraph_inputs = str(len(list_nodes_updated))
    dictionary["num_subgraph_inputs"] = f"{num_subgraph_inputs}"
    list_heads = dictionary["heads"]
    list_heads_updated = []
    for iterator in list_heads:
        if dictionary["nodes"][iterator[0]]["op"] != "const":
            if iterator[0] != 0:
                get_index = dictionary["nodes"][iterator[0]]["attrs"]["mrvl_nodes_idx"][0]
                new_index = get_index + num_in - 1
                list_heads_updated.append([new_index, 0, 0])
    dictionary["heads"] = list_heads_updated

    num_subgraph_outputs = str(len(list_heads_updated))
    dictionary["num_subgraph_outputs"] = f"{num_subgraph_outputs}"

    # Delete the constant nodes, these are not required for the constants file
    dictionary["nodes"] = [
        feature for feature in dictionary["nodes"] if "const" not in feature["op"]
    ]

    # Remove un-needed array nesting
    for iterator in dictionary["nodes"]:
        if iterator["op"] not in "input":
            for it2 in iterator["attrs"]:
                if it2 not in [
                    "num_inputs",
                    "num_outputs",
                    "mrvl_nodes_idx",
                    "mean_const",
                    "var_const",
                    "beta_const",
                    "kernel_const",
                    "bias_const",
                    "gamma_const",
                    "input_const",
                ]:
                    iterator["attrs"][it2] = iterator["attrs"][it2][0]

    # Now create the dltype and dlshape attributes
    dltype = ["list_str"]
    shape = ["list_shape"]
    list_types = []
    list_shapes = []
    for iterator in dictionary["nodes"]:
        list_types.append(iterator["attrs"]["dtype"][0])
        list_shapes.append(iterator["attrs"]["shape"][0])
    dltype.append(list_types)
    shape.append(list_shapes)
    dict_shape_type = {}
    dict_shape_type["shape"] = shape
    dict_shape_type["dltype"] = dltype
    dictionary["attrs"] = dict_shape_type

    nodes_json_string = json.dumps(dictionary)
    return nodes_json_string


@tvm._ffi.register_func("tvm.mrvl.ModifyConstNames")
def modify_const_names(nodes_json_str, consts_json_str):
    """This takes the graph module returned by relay.build an generates nodes and constant
       meta data suitable for compilation by the back end.

    Parameters
    ----------
    nodes_json_str: string
        The nodes json string suitable for the Marvell backend.

    consts_json_str: string
        The consts_json_string generated by the backend compiler.

    Returns
    -------
    modified_nodes_consts: string
        This returns a concatenated string of the nodes_json and modified
        consts json file, seperated by a delimiter |. The modification to the
        consts file is necessary since we have added the Merge Compiler Pass
        which names the constants in a form unsuitable for the backend.
    """

    nodes = json.loads(nodes_json_str)
    const = json.loads(consts_json_str)
    for iterator in nodes["nodes"]:
        hasBias = False
        for attrs in iterator["attrs"]:
            if attrs == "bias_const_name":
                hasBias = True
        for attrs in iterator["attrs"]:
            if attrs == "kernel_const_name":
                new_name = iterator["name"] + "_const_0"
                const[new_name] = const.pop(iterator["attrs"][attrs][0])
                iterator["attrs"][attrs][0] = new_name
                map_kernel = {}
                map_kernel["shape"] = const[new_name]["shape"]
                map_kernel["dtype"] = const[new_name]["dtype"]
                map_kernel["min"] = const[new_name]["min"]
                map_kernel["max"] = const[new_name]["max"]
                map_kernel["name"] = new_name
                iterator["attrs"]["kernel_const"] = map_kernel
            if attrs == "bias_const_name":
                new_name = iterator["name"] + "_const_1"
                const[new_name] = const.pop(iterator["attrs"][attrs][0])
                iterator["attrs"][attrs][0] = new_name
                bias_map = {}
                bias_map["shape"] = const[new_name]["shape"]
                bias_map["dtype"] = const[new_name]["dtype"]
                bias_map["min"] = const[new_name]["min"]
                bias_map["max"] = const[new_name]["max"]
                bias_map["name"] = new_name
                iterator["attrs"]["bias_const"] = bias_map
            if attrs == "gamma_const_name":
                if hasBias:
                    new_name = iterator["name"] + "_const_2"
                else:
                    new_name = iterator["name"] + "_const_1"
                const[new_name] = const.pop(iterator["attrs"][attrs][0])
                iterator["attrs"][attrs][0] = new_name
                gamma_map = {}
                gamma_map["shape"] = const[new_name]["shape"]
                gamma_map["dtype"] = const[new_name]["dtype"]
                gamma_map["name"] = new_name
                iterator["attrs"]["gamma_const"] = gamma_map
            if attrs == "beta_const_name":
                if hasBias:
                    new_name = iterator["name"] + "_const_3"
                else:
                    new_name = iterator["name"] + "_const_2"
                const[new_name] = const.pop(iterator["attrs"][attrs][0])
                iterator["attrs"][attrs][0] = new_name
                beta_map = {}
                beta_map["shape"] = const[new_name]["shape"]
                beta_map["dtype"] = const[new_name]["dtype"]
                beta_map["name"] = new_name
                iterator["attrs"]["beta_const"] = beta_map
            if attrs == "mean_const_name":
                if hasBias:
                    new_name = iterator["name"] + "_const_4"
                else:
                    new_name = iterator["name"] + "_const_3"
                const[new_name] = const.pop(iterator["attrs"][attrs][0])
                iterator["attrs"][attrs][0] = new_name
                mean_map = {}
                mean_map["shape"] = const[new_name]["shape"]
                mean_map["dtype"] = const[new_name]["dtype"]
                mean_map["name"] = new_name
                iterator["attrs"]["mean_const"] = mean_map
            if attrs == "var_const_name":
                if hasBias:
                    new_name = iterator["name"] + "_const_5"
                else:
                    new_name = iterator["name"] + "_const_4"
                const[new_name] = const.pop(iterator["attrs"][attrs][0])
                iterator["attrs"][attrs][0] = new_name
                var_map = {}
                var_map["shape"] = const[new_name]["shape"]
                var_map["dtype"] = const[new_name]["dtype"]
                var_map["name"] = new_name
                iterator["attrs"]["var_const"] = var_map
            if attrs == "input_const_name":
                new_name = iterator["name"] + "_const_0"
                const[new_name] = const.pop(iterator["attrs"][attrs][0])
                const[new_name]["shape"] = list(map(int, iterator["attrs"]["input_const_shape"]))
                iterator["attrs"][attrs][0] = new_name
                map_const = {}
                map_const["shape"] = const[new_name]["shape"]
                map_const["dtype"] = const[new_name]["dtype"]
                map_const["min"] = const[new_name]["min"]
                map_const["max"] = const[new_name]["max"]
                map_const["name"] = new_name
                iterator["attrs"]["input_const"] = map_const

    nodes_mod_str = json.dumps(nodes, indent=2)
    const_mod_str = json.dumps(const, indent=2)
    return nodes_mod_str + "|" + const_mod_str


def get_working_dir():
    """Obtain the current working directory from where tvm is invoked"""
    return os.getcwd()


@tvm._ffi.register_func("tvm.mrvl.WriteJsonFile")
def write_json_file(json_string, json_filename):
    """Generate json file under working directory"""
    working_dir = get_working_dir()
    json_file = os.path.join(working_dir, json_filename)
    with open(json_file, "w") as out_file:
        out_file.write(json_string)
    return json_file


def delete_temp_files(symbol_name):
    """Delete temporary files generated by the Marvell compiler"""
    working_dir = get_working_dir()
    nodes_json_file = os.path.join(working_dir, f"{symbol_name}-nodes.json")
    consts_json_file = os.path.join(working_dir, f"{symbol_name}-consts.json")
    os.remove(nodes_json_file)
    os.remove(consts_json_file)
    bin_folder = os.path.join(working_dir, "bin_" + symbol_name)
    if "MRVL_SAVE_MODEL_BIN" not in os.environ:
        shutil.rmtree(bin_folder)


@tvm._ffi.register_func("tvm.mrvl.CompileModel")
def compile_model(
    symbol_name,
    nodes_json_string,
    consts_json_string,
    compiler_opts,
):
    """Compile the model using Marvell Backend compiler and return the generated binary"""
    # generate pair of json files
    nodes_json_file = write_json_file(nodes_json_string, f"{symbol_name}-nodes.json")
    consts_json_file = write_json_file(consts_json_string, f"{symbol_name}-consts.json")
    mrvl_exec = "mrvl-tmlc"
    exec_on_path = shutil.which(mrvl_exec)
    if exec_on_path is None:
        error_msg = (
            "Marvell Compiler not found! Please specify the path to Marvell tools "
            "by adding it to $PATH."
        )
        raise RuntimeError(error_msg)

    # Parse the nodes_json string for the batch size
    dictionary = json.loads(nodes_json_string)
    batch_size = dictionary["batch_size"]

    # Check for supported batch size
    if int(batch_size) > 8:
        error_msg = "Compilation ERROR: mrvl-tmlc supports batch_size <= 8"
        raise RuntimeError(error_msg)

    # Invoke Marvell Backend with appropriate options
    compile_cmd = (
        mrvl_exec
        + " -mn "
        + symbol_name
        + " -f1 "
        + nodes_json_file
        + " -f2 "
        + consts_json_file
        + " "
        + compiler_opts
        + " -b "
        + batch_size
    )

    ret_val = os.system(compile_cmd)
    if ret_val == 0:
        # Read generated binary and encode in base64 format
        working_dir = get_working_dir()
        bin_file = os.path.join(working_dir, "bin_" + symbol_name, symbol_name + ".bin")

        with open(bin_file, "rb") as f:
            data = bytearray(f.read())
            base64_bytes = base64.b64encode(data)
            if not data:
                raise RuntimeError("Compilation ERROR: Marvell binary could not be generated")
            # Cleanup Temporary Files
            delete_temp_files(symbol_name)
            return base64_bytes
    else:
        error_msg = "Compilation ERROR: Error compiling Marvell region!"
        raise RuntimeError(error_msg)


@tvm._ffi.register_func("tvm.mrvl.CleanUpSim")
def clean_up_sim(bin_file, input_json, input_bin, out_bin_prefix, num_outputs):
    os.remove(bin_file)
    os.remove(input_json)
    os.remove(input_bin)
    for i in range(num_outputs):
        out_bin = out_bin_prefix + "-" + str(i) + ".bin"
        os.remove(out_bin)


@tvm._ffi.register_func("tvm.mrvl.SearchPath")
def search_path(file_name):
    path = shutil.which(file_name)
    if path is None:
        return ""
    return os.path.dirname(path)


@tvm._ffi.register_func("tvm.mrvl.JsonToBin")
def convert_json_to_bin(json_file, input_bin_file):
    with open(json_file) as input_json:
        data = json.load(input_json)
        data_float = np.array(data["inputs"], dtype=np.float32)
        data_b = data_float.tobytes()
        with open(input_bin_file, "wb") as f:
            f.write(data_b)


@tvm._ffi.register_func("tvm.mrvl.RunSim")
def run_simulation(run_command, sim_directory):
    cwd_path = get_working_dir()
    os.mkdir(sim_directory)
    os.chdir(sim_directory)
    os.system(run_command)
    os.chdir(cwd_path)
    shutil.rmtree(sim_directory)


@tvm._ffi.register_func("tvm.mrvl.TempDir")
def get_temp_dir():
    return tempfile.gettempdir()
