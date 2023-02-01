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
"""Local builder for microTVM projects that compile on the local host"""

import os
import tempfile
from typing import Optional, Dict
from tvm.ir import IRModule
from tvm.runtime import NDArray
from tvm.target import Target
from tvm.meta_schedule.builder import LocalBuilder
from tvm.driver.build_module import OperatorModule
from tvm import micro
from tvm.contrib.tar import tar
from tvm.relay.backend import Runtime
from tvm.driver import build as tvm_build
from tvm.tir.transform import RemoveWeightLayoutRewriteBlock


def get_local_builder_micro():
    """Return micro-compatible Builder for meta schedule."""

    def _micro_build(
        mod: IRModule, target: Target, _params: Optional[Dict[str, NDArray]]
    ) -> OperatorModule:
        """Build function for micro targets.

        Parameters
        ----------
        mod : IRModule
            The IRModule to be built.
        target : Target
            The target to be built.
        _params : Optional[Dict[str, NDArray]]
            The parameters to be used for the build. Must be None.

        Returns
        -------
        rt_mod : OperatorModule
            The built Module.
        """

        # Note: tvm_build assigns "global_symbol" to the name of generated C function
        # changing it is necessary for micro targets,
        # since the generated projects already include a main function.
        prim_func = mod["main"].with_attr("global_symbol", "default_function")
        mod = IRModule({"main": prim_func})
        runtime = Runtime("crt", {"system-lib": True})
        mod = RemoveWeightLayoutRewriteBlock(skip_ndarray_rewrite=True)(mod)
        rt_mod = tvm_build(mod, target=target, runtime=runtime)
        return rt_mod

    def _micro_export(mod: OperatorModule) -> str:
        """Export function for micro targets.

        Parameters
        ----------
        mod : OperatorModule
            The Module to be exported.

        Returns
        -------
        artifact_path : str
            The path to the exported Module.
        """
        artifact_path = os.path.join(tempfile.mkdtemp(), "tvm_tmp_mod." + tar.output_format)
        micro.export_model_library_format(mod, artifact_path)
        return artifact_path

    return LocalBuilder(f_build=_micro_build, f_export=_micro_export)
