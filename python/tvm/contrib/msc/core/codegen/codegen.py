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

import os
import subprocess
from typing import Dict, List, Optional, Any, Callable

import tvm
from tvm import relax
from tvm.relax import PyExprVisitor
from tvm.contrib.msc.core import transform as msc_transform
from tvm.contrib.msc.core.ir import MSCGraph, MSCTensor
from tvm.contrib.msc.core.frontend import from_relay
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
    coda_format: str
        The code format cpp| python.
    """

    def __init__(
        self,
        graph: MSCGraph,
        source_getter: Callable[[MSCGraph, str, str], str],
        codegen_config: Optional[Dict[str, str]] = None,
        print_config: Optional[Dict[str, str]] = None,
        build_folder: msc_utils.MSCDirectory = None,
        code_format: str = "python",
    ):
        self._graph = graph
        self._source_getter = source_getter
        self._codegen_config = msc_utils.dump_dict(codegen_config)
        self._print_config = msc_utils.dump_dict(print_config)
        self._build_folder = build_folder or msc_utils.msc_dir(keep_history=False, cleanup=True)
        self._code_format = code_format

    def load(
        self,
        inputs: Optional[List[Any]] = None,
        pre_load: Optional[Callable[[msc_utils.MSCDirectory], Any]] = None,
        post_load: Optional[Callable[[Any, msc_utils.MSCDirectory], Any]] = None,
        build_model: bool = True,
    ) -> Any:
        """Generate source and load the model

        Parameters
        -------
        inputs: list<any>
            The inputs to build the model.
        pre_load: Callable
            The pre processing method before load.
        post_load: Callable
            The post processing method after load.
        build_model: bool
            Whether to build the model.

        Returns
        -------
        obj: model object
            The model object for the framework.
        """

        sources = self._source_getter(self._graph, self._codegen_config, self._print_config)
        inputs = inputs or []
        with self._build_folder as folder:
            # pre processing
            if pre_load:
                pre_load(folder)
            for name, source in sources.items():
                folder.add_file(name, source)
            if build_model:
                if self._code_format == "cpp":
                    with folder.create_dir("build"):
                        command = "cmake ../ && make && mv {} ../".format(self._graph.name)
                        with open("codegen.log", "w") as log_f:
                            process = subprocess.Popen(
                                command, stdout=log_f, stderr=log_f, shell=True
                            )
                        process.wait()
                        assert (
                            process.returncode == 0
                        ), "Failed to build {} under {}, check codegen.log for detail".format(
                            self._graph.name, os.getcwd()
                        )
                    obj = self._graph.name
                elif self._code_format == "python":
                    builder = msc_utils.load_callable(self._graph.name + ".py:" + self._graph.name)
                    obj = builder(*inputs)
                else:
                    raise NotImplementedError(
                        "Code format {} is not supported".format(self._code_format)
                    )
                # post processing
                if post_load:
                    obj = post_load(obj, folder)
            else:
                obj = None
        return obj


def to_relax(
    graph: MSCGraph,
    weights: Optional[Dict[str, tvm.nd.array]] = None,
    codegen_config: Optional[Dict[str, str]] = None,
    print_config: Optional[Dict[str, str]] = None,
    build_folder: msc_utils.MSCDirectory = None,
    plugin: Any = None,
    use_alias: bool = True,
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
    use_alias: bool
        Whether to use alias for input.

    Returns
    -------
    mod: IRModule
        The IRModule of relax.
    """

    @relax.expr_functor.visitor
    class NamesGetter(PyExprVisitor):
        """Visitor for get attributes in span"""

        def get_names(self, expr: relax.Expr) -> dict:
            self._names = {}
            if isinstance(expr, relax.Expr):
                self.visit_expr(expr)
            elif isinstance(expr, relax.BindingBlock):
                self.visit_binding_block(expr)
            return self._names

        def visit_var_binding_(self, binding: relax.VarBinding) -> None:
            super().visit_var_binding_(binding)
            self._names[binding.var.name_hint] = binding.var.name_hint

    def _to_var(tensor: MSCTensor):
        v_name = tensor.alias if use_alias else graph.find_producer(tensor).name
        dims = [
            d if isinstance(d, int) else tvm.tir.Var(d, "int64") for d in tensor.get_shape(True)
        ]
        return tvm.relax.Var(v_name, tvm.relax.TensorStructInfo(dims, tensor.dtype_name))

    def _save_weights(folder: msc_utils.MSCDirectory):
        if weights:
            with open(folder.relpath(graph.name + "_params.bin"), "wb") as f_params:
                f_params.write(tvm.runtime.save_param_dict(weights))

    # pylint: disable=unused-argument
    def _post_proc(mod: tvm.IRModule, folder: msc_utils.MSCDirectory) -> tvm.IRModule:
        passes, var_names = [], NamesGetter().get_names(mod["main"])
        if weights:
            passes.append(msc_transform.BindNamedParams("main", weights))
        # The canonicalization of relax variable bindings is not required
        # for correctness.  It does, however, remove trivial `x = y`
        # bindings, preventing test cases from depending on their
        # presence.
        passes.extend(
            [
                msc_transform.SetExprName(var_names=var_names),
                tvm.relax.transform.CanonicalizeBindings(),
                tvm.relax.transform.ConvertToDataflow(min_size=1),
            ]
        )
        return tvm.ir.transform.Sequential(
            passes, name="tvm.contrib.msc.core.codegen.to_relax_postproc"
        )(mod)

    source_getter = tvm.get_global_func("msc.framework.tvm.GetRelaxSources")
    codegen = CodeGen(graph, source_getter, codegen_config, print_config, build_folder)
    model_args = [_to_var(i) for i in graph.get_inputs()]
    if plugin:
        model_args = model_args + [plugin]
    return codegen.load(model_args, pre_load=_save_weights, post_load=_post_proc)


def relay_to_relax(
    relay_mod: tvm.IRModule,
    params: Optional[Dict[str, tvm.nd.array]] = None,
    trans_config: Optional[Dict[str, str]] = None,
    build_config: Optional[Dict[str, str]] = None,
    opt_config: Optional[Dict[str, str]] = None,
    build_folder: msc_utils.MSCDirectory = None,
) -> tvm.IRModule:
    """Change relay IRModule to relax MSCGraph.

    Parameters
    ----------
    relay_mod: IRModule
        The IRModule of relay.
    params: dict of <string:tvm.ndarray>
        The parameters of the IRModule.
    trans_config: dict
        The config for transform IRModule.
    build_config: dict
        The config for build MSCGraph.
    opt_config: dict
        The config for optimize the relay before translate.
    build_folder: MSCDirectory
        The folder for saving scripts and datas.

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

    return to_relax(graph, weights, codegen_config={"from_relay": True}, build_folder=build_folder)
