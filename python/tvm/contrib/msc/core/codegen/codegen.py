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
"""tvm.contrib.msc.core.codegen.codegen"""

from typing import Dict, List, Optional, Any, Callable

import tvm
from tvm.relax.transform import BindParams
from tvm.contrib.msc.core.ir import MSCGraph
from tvm.contrib.msc.core.ir.translate import from_relay
from tvm.contrib.msc.core import utils as msc_utils


class CodeGen(object):
    """Manager class to generate codes and load model

    Parameters
    ----------
    graph: MSCGraph
        The reference graph for codegen.
    source_getter: Callable
        The method to get sources.
    codegen_config: dict<string, string>
        The config to generate code.
    print_config: dict<string, string>
        The config to print code.
    build_folder: MSCDirectory
        The codegen folder.
    """

    def __init__(
        self,
        graph: MSCGraph,
        source_getter: Callable[[MSCGraph, str, str], str],
        codegen_config: Optional[Dict[str, str]] = None,
        print_config: Optional[Dict[str, str]] = None,
        build_folder: msc_utils.MSCDirectory = None,
    ):
        self._graph = graph
        self._source_getter = source_getter
        self._codegen_config = msc_utils.dump_dict(codegen_config)
        self._print_config = msc_utils.dump_dict(print_config)
        self._build_folder = build_folder or msc_utils.msc_dir(keep_history=False, cleanup=True)

    def load(
        self,
        inputs: Optional[List[Any]] = None,
        weights_binder: Optional[Callable[[MSCGraph, Any, msc_utils.MSCDirectory], Any]] = None,
    ) -> Any:
        """Generate source and load the model

        Parameters
        -------
        inputs: list<any>
            The inputs to build the model.
        weights_binder: Callable
            The method for binding weights to the model.

        Returns
        -------
        obj: model object
            The model object for the framework.
        """

        sources = self._source_getter(self._graph, self._codegen_config, self._print_config)
        inputs = inputs or []
        with self._build_folder as folder:
            for name, source in sources.items():
                folder.add_file(name, source)
            builder = msc_utils.load_callable(self._graph.name + ".py:" + self._graph.name)
            obj = builder(*inputs)
            # load weights
            if weights_binder:
                obj = weights_binder(obj, folder)
        return obj


def relay_to_relax(
    relay_mod: tvm.IRModule,
    params: Optional[Dict[str, tvm.nd.array]] = None,
    trans_config: Optional[Dict[str, str]] = None,
    build_config: Optional[Dict[str, str]] = None,
    opt_config: Optional[Dict[str, str]] = None,
) -> tvm.IRModule:
    """Change IRModule to MSCGraph.

    Parameters
    ----------
    relay_mod: IRModule
        The IRModule of relay.
    params: dict of <string:tvm.ndarray>
        The parameters of the IRModule.
    trans_config: dict
        The config for transfrorm IRModule.
    build_config: dict
        The config for build MSCGraph.
    opt_config: dict
        The config for optimize the relay before translate.

    Returns
    -------
    relax_mod: IRModule
        The IRModule of relax.
    """

    graph, weights = from_relay(
        relay_mod,
        params,
        trans_config=trans_config,
        build_config=build_config,
        opt_config=opt_config,
    )
    source_getter = tvm.get_global_func("msc.framework.tvm.GetRelaxSources")
    codegen_config = {"from_relay": True}
    codegen = CodeGen(graph, source_getter, codegen_config)
    inputs = [
        tvm.relax.Var(i.alias, tvm.relax.TensorStructInfo(i.get_shape(), i.dtype_name))
        for i in graph.get_inputs()
    ]

    # pylint: disable=unused-argument
    def _bind_weights(mod: tvm.IRModule, folder: msc_utils.MSCDirectory) -> tvm.IRModule:
        return BindParams("main", weights)(mod)

    return codegen.load(inputs, _bind_weights)
