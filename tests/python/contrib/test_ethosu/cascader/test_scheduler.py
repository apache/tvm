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
import pytest

import tvm.contrib.ethosu.cascader as cs

from .infra import ethosu_enabled

if ethosu_enabled:

    def test_cascade(
        SRAM, FLASH, TwoConv2DWithSliceTE, TwoConv2DTE, MobileNetv1StartTE, MobileNetv1TE
    ):
        fixtures = [
            TwoConv2DTE,
            TwoConv2DWithSliceTE,
            MobileNetv1StartTE,
            MobileNetv1TE,
        ]
        device_config = cs.EthosuDeviceConfig("ethos-u55-256")
        for sch, te_graph, const_dict in fixtures:
            options = cs.CascaderOptions(
                cascade_region=SRAM,
                max_proposals=64,
                stripe_factors=4,
                max_plan_size=10,
                max_open_plans=8,
                max_closed_plans=32,
                always_copy_size=1024,
                disable_pareto_plans=False,
                disable_pareto_proposals=False,
            )
            cs.cascade(sch, te_graph, const_dict, options, SRAM, FLASH, [SRAM], device_config)


if __name__ == "__main__":
    pytest.main([__file__])
