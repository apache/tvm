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
# pylint: disable=invalid-name, unused-argument
"""Backend compiler related feature registration"""
from __future__ import absolute_import

import topi
from .. import op as reg
from ..op import OpPattern


# adaptive_max_pool2d
@reg.register_schedule("contrib.adaptive_max_pool2d")
def schedule_adaptive_max_pool2d(_, outs, target):
    """Schedule definition of adaptive_max_pool2d"""
    with target:
        return topi.generic.schedule_adaptive_pool(outs)

reg.register_pattern("contrib.adaptive_max_pool2d", OpPattern.OUT_ELEMWISE_FUSABLE)


# adaptive_avg_pool2d
@reg.register_schedule("contrib.adaptive_avg_pool2d")
def schedule_adaptive_avg_pool2d(_, outs, target):
    """Schedule definition of adaptive_avg_pool2d"""
    with target:
        return topi.generic.schedule_adaptive_pool(outs)

reg.register_pattern("contrib.adaptive_avg_pool2d", OpPattern.OUT_ELEMWISE_FUSABLE)
