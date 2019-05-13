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
"""Helper functions and global data"""


RULE_OUT_NODE_NAMES = ["Tuple", "TupleGetItem", "batch_flatten", "transpose", "reshape",
                       "multibox_prior", "multibox_transform_loc", "where",
                       "non_max_suppression", "strided_slice"]

# We set a large time to represent an invalid layout-transformation.
# This number is set to be 10e9 seconds to align with autotvm.
INVALID_LAYOUT_TIME = 10e9
