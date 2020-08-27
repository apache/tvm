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
# pylint: disable=invalid-name,unused-variable,unused-argument,no-member
"""Arm target utility functions"""

import tvm
def is_fast_int8_on_arm():
    """ Checks whether the hardware has support for fast Int8 arithmetic operations. """
    target = tvm.target.Target.current(allow_none=False)
    return "+v8.2a" in target.mattr and "+dotprod" in target.mattr

def is_aarch64_arm():
    """ Checks whether we are compiling for an AArch64 target. """
    target = tvm.target.Target.current(allow_none=False)
    return 'aarch64' in target.attrs.get("mtriple", "")
