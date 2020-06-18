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
"""Definition of vision ops"""
from __future__ import absolute_import

from .. import op as reg
from .. import strategy
from ..op import OpPattern

# multibox_prior
reg.register_strategy("vision.multibox_prior", strategy.multibox_prior_strategy)
reg.register_pattern("vision.multibox_prior", OpPattern.OPAQUE)


# multibox_transform_loc
reg.register_strategy("vision.multibox_transform_loc", strategy.multibox_transform_loc_strategy)
reg.register_pattern("vision.multibox_transform_loc", OpPattern.OPAQUE)


# Get counts of valid boxes
reg.register_strategy("vision.get_valid_counts", strategy.get_valid_counts_strategy)
reg.register_pattern("vision.get_valid_counts", OpPattern.OPAQUE)


# non-maximum suppression
reg.register_strategy("vision.non_max_suppression", strategy.nms_strategy)
reg.register_pattern("vision.non_max_suppression", OpPattern.OPAQUE)
