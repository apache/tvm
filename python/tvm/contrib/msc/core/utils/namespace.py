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
"""tvm.contrib.msc.core.utils.namespace"""

from typing import Any, Optional
import copy


class MSCMap:
    """Global Namespace map for MSC"""

    MAP = {}

    @classmethod
    def set(cls, key: str, value: Any):
        cls.MAP[key] = value

    @classmethod
    def get(cls, key: str, default: Optional[Any] = None):
        return cls.MAP.get(key, default)

    @classmethod
    def clone(cls, key: str, default: Optional[Any] = None):
        return copy.deepcopy(cls.get(key, default))

    @classmethod
    def delete(cls, key: str):
        if key in cls.MAP:
            return cls.MAP.pop(key)
        return None

    @classmethod
    def contains(cls, key: str):
        return key in cls.MAP

    @classmethod
    def reset(cls):
        cls.MAP = {}


class MSCKey:
    """Keys for the MSCMap"""

    WORKSPACE = "workspace"
    VERBOSE = "verbose"
    GLOBALE_LOGGER = "global_logger"
    MSC_STAGE = "msc_stage"
    TIME_STAMPS = "time_stamps"

    PRUNERS = "pruners"
    QUANTIZERS = "quantizers"
    DISTILLERS = "distillers"
    TRACKERS = "trackers"

    FUSED_CNT = "fused_cnt"


class MSCFramework:
    """Framework type for the MSC"""

    MSC = "msc"
    TVM = "tvm"
    TORCH = "torch"
    TENSORFLOW = "tensorflow"
    TENSORRT = "tensorrt"
