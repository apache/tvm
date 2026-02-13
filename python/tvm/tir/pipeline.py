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

# pylint: disable=invalid-name
"""The TIR backend compilation pipeline."""

import tvm
from tvm import tir


def finalize_host_passes():  # pylint: disable=unused-argument
    """The default finalization passes for TIR backend."""
    host_pass_list = [
        tir.transform.LowerTVMBuiltin(),
        tir.transform.LowerCustomDatatypes(),
        tir.transform.LowerIntrin(),
    ]
    return tvm.ir.transform.Sequential(host_pass_list)


def finalize_device_passes():  # pylint: disable=unused-argument
    """The default finalization passes for TIR backend."""
    device_pass_list = [
        tir.transform.LowerWarpMemory(),
        tir.transform.Simplify(),
        tir.transform.LowerCustomDatatypes(),
        tir.transform.LowerIntrin(),
    ]
    return tvm.ir.transform.Sequential(device_pass_list)


# global map of pre-built pipelines
PIPELINE_MAP = {}


def get_tir_pipeline(name: str = None, **kwargs) -> tvm.transform.Pass:
    """Get pre-build pipeline by name

    Parameters
    ----------
    name : Optional[str]
        Name of the pipeline
    """
    if name == "default":
        # for now, defualt to s_tir pipeline
        name = "s_tir"
    if name not in PIPELINE_MAP:
        raise ValueError(
            f"Unknown pre-built pipeline {name}," f"candidates are {list(PIPELINE_MAP.keys())}"
        )
    return PIPELINE_MAP[name](**kwargs)


def get_default_tir_pipeline(
    target: tvm.target.Target,  # pylint: disable=unused-argument
) -> tvm.transform.Pass:
    """Get the default TIR pipeline for the given target."""
    if target.kind.name == "opencl" and "adreno" in target.keys:
        return get_tir_pipeline("adreno")
    else:
        return get_tir_pipeline("s_tir")
