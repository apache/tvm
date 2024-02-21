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
"""tvm.contrib.msc.framework.tensorflow.codegen.codegen"""

from typing import Dict, Optional, Any

import tvm
from tvm.contrib.msc.core.ir import MSCGraph
from tvm.contrib.msc.core.codegen import CodeGen
from tvm.contrib.msc.core import utils as msc_utils
from tvm.contrib.msc.framework.tensorflow import tf_v1
from tvm.contrib.msc.framework.tensorflow import _ffi_api


def to_tensorflow(
    graph: MSCGraph,
    weights: Optional[Dict[str, tvm.nd.array]] = None,
    codegen_config: Optional[Dict[str, str]] = None,
    print_config: Optional[Dict[str, str]] = None,
    build_folder: msc_utils.MSCDirectory = None,
    plugin: Any = None,
) -> tf_v1.Graph:
    """Change MSCGraph to tensorflow graph.

    Parameters
    ----------
    graph: tvm.contrib.msc.core.ir.MSCGraph
        The translated graph.
    weights: dict of <string:tvm.ndarray>
        The parameters of the IRModule.
    codegen_config: dict
        The config for codegen.
    print_config: dict
        The config for print.
    build_folder: MSCDirectory
        The folder for saving scripts and datas.
    plugin: PluginManager
        The plugin manager.

    Returns
    -------
    tf_graph: tf_v1.Graph
        The tensorflow Graph.
    """

    def _save_weights(folder: msc_utils.MSCDirectory):
        if weights:
            with open(folder.relpath(graph.name + "_params.bin"), "wb") as f_params:
                f_params.write(tvm.runtime.save_param_dict(weights))

    inputs = [tf_v1.placeholder(i.dtype_name, i.get_shape(), i.alias) for i in graph.get_inputs()]
    codegen = CodeGen(
        graph, _ffi_api.GetTensorflowSources, codegen_config, print_config, build_folder
    )
    model_args = inputs + [weights]
    if plugin:
        model_args = model_args + [plugin]
    return codegen.load(model_args, pre_load=_save_weights)
