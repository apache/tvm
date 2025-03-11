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

# pylint: disable=unused-argument
"""tvm.contrib.msc.framework.torch.frontend.translate"""

from typing import Dict, Optional, Tuple, List, Union

import tvm

from tvm.contrib.msc.core.ir.graph import MSCGraph
from tvm.contrib.msc.framework.tensorflow import tf_v1


def from_tensorflow(
    graph_def: tf_v1.GraphDef,
    shape_dict: Dict[str, List[int]],
    outputs: List[str],
    trans_config: Optional[Dict[str, str]] = None,
    build_config: Optional[Dict[str, str]] = None,
    opt_config: Optional[Dict[str, str]] = None,
    as_msc: bool = True,
) -> Tuple[Union[MSCGraph, tvm.IRModule], Dict[str, tvm.nd.array]]:
    """Change tensorflow GraphDef to MSCGraph.

    Parameters
    ----------
    graph_def: tf_v1.GraphDef
        The graph define of tensorflow.
    shape_dict: dict<str,list<int>>
        The shape dict of inputs.
    outputs: list<str>
        The output names.
    trans_config: dict
        The config for transform IRModule.
    build_config: dict
        The config for build MSCGraph.
    opt_config: dict
        The config for optimize before translate.
    as_msc: bool
        Set to to return msc graph, otherwise relax mod

    Returns
    -------
    graph/mod: tvm.contrib.msc.core.ir.MSCGraph/tvm.IRModule
        The translated graph/IRModule.
    weights: dict of <string:tvm.ndarray>
        The weights from the IRModule.
    """

    raise NotImplementedError("translate relax module from tensorflow is not implemented")
