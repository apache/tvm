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
"""Configuration about tests"""
from __future__ import absolute_import as _abs

import os
import tvm

def ctx_list():
    """Get context list for testcases"""
    device_list = os.environ.get("NNVM_TEST_TARGETS", "")
    device_list = (device_list.split(",") if device_list
                   else ["llvm", "cuda"])
    device_list = set(device_list)
    res = [(device, tvm.context(device, 0)) for device in device_list]
    return [x for x in res if x[1].exist]
