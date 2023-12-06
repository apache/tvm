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
"""tvm.contrib.msc.core.tools.prune.method"""

from typing import List
import numpy as np

from tvm.contrib.msc.core.tools.tool import ToolType, BaseTool
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils


class PruneMethod(object):
    """Default prune method"""

    @classmethod
    def prune_axis(cls, data: np.ndarray, axis: int, indices: List[int]) -> np.ndarray:
        """Delete indices on axis

        Parameters
        ----------
        data: np.ndarray
            The source data.
        axis: int
            The axis to prune
        indices: list<int>
            The indices to be pruned

        Returns
        -------
        data: np.ndarray
            The pruned data.
        """

        left_datas = [
            d for idx, d in enumerate(np.split(data, data.shape[axis], axis)) if idx in indices
        ]
        return np.concatenate(left_datas, axis=axis)

    @classmethod
    def per_channel(
        cls,
        pruner: BaseTool,
        data: np.ndarray,
        name: str,
        consumer: str,
        in_axis: int,
        out_axis: int,
        in_indices: List[int],
        density: float,
        stride: int = 8,
    ) -> np.ndarray:
        """Prune the data

        Parameters
        ----------
        pruner: BasePruner
            The pruner
        data: np.ndarray
            The source data.
        name: str
            The name of the weight.
        consumer: str
            The name of the consumer.
        in_axis: int
            The input axis
        out_axis: int
            The output axis
        in_indices: list<int>
            The input indices to be pruned
        density: float
            The density to prune
        stride: int
            The prune stride

        Returns
        -------
        plan: dict
            The plan of the tensor.
        """

        config = {"in_indices": in_indices, "out_indices": []}
        if density == 1:
            return config
        if len(in_indices) > 0:
            data = cls.prune_axis(data, in_axis, in_indices)
        out_dim = data.shape[out_axis]
        left_num = int(((density * out_dim + stride) // stride) * stride)
        axis_sum = [np.abs(d).sum() for d in np.split(data, out_dim, out_axis)]
        rank = np.argsort(np.array(axis_sum))
        config["out_indices"] = rank[-left_num:].tolist()
        return config

    @classmethod
    def framework(cls):
        return MSCFramework.MSC

    @classmethod
    def tool_type(cls):
        return ToolType.PRUNER


msc_utils.register_tool_method(PruneMethod)
