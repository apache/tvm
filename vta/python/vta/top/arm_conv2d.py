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
"""Reuse conv2d schedule from ARM CPU"""

import tvm

from topi.nn import conv2d, conv2d_alter_layout
from topi import generic

@conv2d.register(["vtacpu", "vta"])
def compute(*args, **kwargs):
    with tvm.target.arm_cpu("vtacpu"):
        return conv2d(*args, **kwargs)

@generic.schedule_conv2d_nchw.register(["vtacpu", "vta"])
def schedule(*args, **kwargs):
    with tvm.target.arm_cpu("vtacpu"):
        return generic.schedule_conv2d_nchw(*args, **kwargs)

@conv2d_alter_layout.register(["vtacpu", "vta"])
def alter(*args, **kwargs):
    with tvm.target.arm_cpu("vtacpu"):
        return conv2d_alter_layout(*args, **kwargs)
