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
"""tvm.contrib.msc.framework.tvm.codegen.codegen"""

from typing import Dict, Optional, Any

import tvm
from tvm.relax.transform import BindParams
from tvm.contrib.msc.core.ir import MSCGraph
from tvm.contrib.msc.core.codegen import CodeGen
from tvm.contrib.msc.core import utils as msc_utils
from tvm.contrib.msc.framework.tvm import _ffi_api


def to_relax(
    graph: MSCGraph,
    weights: Optional[Dict[str, tvm.nd.array]] = None,
    codegen_config: Optional[Dict[str, str]] = None,
    print_config: Optional[Dict[str, str]] = None,
    build_folder: msc_utils.MSCDirectory = None,
    plugin: Any = None,
) -> tvm.IRModule:
    """Change MSCGraph to IRModule.

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
    mod: IRModule
        The IRModule of relax.
    """

    inputs = [
        tvm.relax.Var(i.alias, tvm.relax.TensorStructInfo(i.get_shape(), i.dtype_name))
        for i in graph.get_inputs()
    ]

    def _save_weights(folder: msc_utils.MSCDirectory):
        if weights:
            with open(folder.relpath(graph.name + "_params.bin"), "wb") as f_params:
                f_params.write(tvm.runtime.save_param_dict(weights))

    # pylint: disable=unused-argument
    def _post_proc(mod: tvm.IRModule, folder: msc_utils.MSCDirectory) -> tvm.IRModule:
        if weights:
            mod = BindParams("main", weights)(mod)
        return tvm.ir.transform.Sequential(
            [
                # The canonicalization of relax variable bindings is not required
                # for correctness.  It does, however, remove trivial `x = y`
                # bindings, preventing test cases from depending on their
                # presence.
                tvm.relax.transform.CanonicalizeBindings(),
                tvm.relax.transform.ConvertToDataflow(min_size=1),
            ],
            name="tvm.contrib.msc.framework.tvm.codegen.to_relax_postproc",
        )(mod)

    codegen = CodeGen(graph, _ffi_api.GetRelaxSources, codegen_config, print_config, build_folder)
    model_args = inputs
    if plugin:
        model_args = model_args + [plugin]
    return codegen.load(model_args, pre_load=_save_weights, post_load=_post_proc)
