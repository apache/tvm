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
"""tvm.contrib.msc.core.tools.track.method"""

from typing import List, Dict
import numpy as np

from tvm.contrib.msc.core.tools.tool import ToolType, BaseTool
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils


class TrackMethod(object):
    """Default track method"""

    @classmethod
    def save_compared(
        cls,
        tracker: BaseTool,
        data: np.ndarray,
        name: str,
        consumer: str,
        compare_to: Dict[str, List[str]],
    ) -> np.ndarray:
        """Compare and save the data

        Parameters
        ----------
        tracker: BaseTracker
            The tracker
        data: np.ndarray
            The source data.
        name: str
            The name of the tensor.
        consumer: str
            The name of the consumer.
        compare_to: dict
            The compare config

        Returns
        -------
        plan: dict
            The plan of the tensor.
        """

        data = msc_utils.cast_array(data)
        config = {"info": msc_utils.inspect_array(data)}
        # save the data
        tracker._saver.save_datas({name: data}, tracker._forward_cnt)
        tracker.debug_tensor(data, name, consumer, "save")
        # compare datas
        if tracker._stage in compare_to:
            diffs = {}
            for stage in compare_to[tracker._stage]:
                if stage in tracker._loaders:
                    if not tracker._loaders[stage].has_data(name, tracker._forward_cnt):
                        continue
                    golden = tracker._loaders[stage].load_data(name, tracker._forward_cnt)
                    report = msc_utils.compare_arrays({name: golden}, {name: data})
                    diff_msg = "{}{} to {} -> {}".format(
                        tracker.msg_mark(), name, stage, report["info"][name]
                    )
                    if report["passed"] == 0:
                        tracker._logger.info(diff_msg)
                    elif tracker.on_debug():
                        tracker._logger.debug(diff_msg)
                    diffs[stage] = {
                        "pass": report["passed"] == 1,
                        "info": msc_utils.inspect_array(np.abs(golden - data)),
                    }
            config["diffs"] = diffs
        return config

    @classmethod
    def framework(cls):
        return MSCFramework.MSC

    @classmethod
    def tool_type(cls):
        return ToolType.TRACKER


msc_utils.register_tool_method(TrackMethod)
