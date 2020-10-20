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

"""Defines functions to work with TVMModule FuncRegistry."""

import json


def graph_json_to_c_func_registry(graph_path, func_registry_path):
    """Convert a graph json file to a CRT-compatible FuncRegistry.

    Parameters
    ----------
    graph_path : str
        Path to the graph JSON file.

    func_registry_path : str
        Path to a .c file which will be written containing the function registry.
    """
    with open(graph_path) as json_f:
        graph = json.load(json_f)

    funcs = []
    for n in graph["nodes"]:
        if n["op"] != "tvm_op":
            continue

        funcs.append(n["attrs"]["func_name"])

    encoded_funcs = f"\\{len(funcs):03o}" + "\\0".join(funcs)
    lines = [
        "#include <tvm/runtime/c_runtime_api.h>",
        "#include <tvm/runtime/crt/module.h>",
        "#include <stdio.h>",
        "",
    ]

    for f in funcs:
        lines.append(
            f"extern int {f}(TVMValue* args, int* type_codes, int num_args, "
            "TVMValue* out_ret_value, int* out_ret_tcode, void* resource_handle);"
        )

    lines.append("static TVMBackendPackedCFunc funcs[] = {")

    for f in funcs:
        lines.append(f"    (TVMBackendPackedCFunc) &{f},")

    lines += [
        "};",
        "static const TVMFuncRegistry system_lib_registry = {",
        f'       "{encoded_funcs}\\0",',
        "        funcs,",
        "};",
        "static const TVMModule system_lib = {",
        "    &system_lib_registry,",
        "};",
        "",
        "const TVMModule* TVMSystemLibEntryPoint(void) {",
        "    return &system_lib;",
        "}",
        "",  # blank line to end the file
    ]
    with open(func_registry_path, "w") as wrapper_f:
        wrapper_f.write("\n".join(lines))
