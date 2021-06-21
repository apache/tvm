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
"""Common utilities for all Android NNAPI partitioning."""
import tvm
from . import transform as _transform


def pre_partition_transform(mod):
    """Perform pre-partition transforms on modules.

    Parameters
    ----------
    mod: tvm.IRModule
        The module to be transformed.

    Returns
    -------
    mod: tvm.IRModule
        The transformed module.
    """
    mod = tvm.relay.transform.ToGraphNormalForm()(mod)
    mod = tvm.relay.transform.RemoveUnusedFunctions()(mod)
    mod = tvm.relay.transform.SimplifyInference()(mod)
    mod = tvm.relay.transform.DeadCodeElimination(inline_once=True)(mod)
    mod = tvm.relay.transform.FoldConstant()(mod)
    mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
    mod = tvm.relay.transform.InferType()(mod)
    mod = _transform.PruneInferenceAgnosticOperators()(mod)
    mod = _transform.TransformRelayOpForNnapi()(mod)
    return mod


def post_partition_transform(
    mod, params, android_nnapi_level=29, external_compiler="android_nnapi"
):
    """Perform post-partition transforms on modules.

    Parameters
    ----------
    mod: tvm.IRModule
        The module to be transformed.

    params: dict of str to tvm.ndarray
        The params dict associated to the module.

    android_nnapi_level: int
        The targeted Android API level.

    external_compiler: str
        The name of the external Relay compiler.

    Returns
    -------
    mod: tvm.IRModule
        The transformed module.

    params: dict of str to NDArray
        The transformed params.
    """
    mod = _transform.AnnotateNnapiFunctionAttributes(
        external_compiler=external_compiler, android_nnapi_level=android_nnapi_level
    )(mod)
    mod, params = _transform.TransformConv2dWeightLayout(
        external_compiler=external_compiler, target_layout="OHWI"
    )(mod, params)
    mod = tvm.relay.transform.LambdaLift()(mod)
    return mod, params
