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
"""Strategies for the my_ai_hw accelerator"""

# Example how to integrate a custom conv1d strategy:

# @relay.op.strategy.override_native_generic_func("custom_conv1d_strategy")
# def custom_conv1d_strategy(attrs, inputs, out_type, target):
#     strategy = _op.OpStrategy()
#     strategy.add_implementation(
#         wrap_compute_conv1d(custom_conv1d_compute),
#         wrap_topi_schedule(custom_conv1d_schedule),
#         name="custom_conv1d.generic",
#     return strategy
#

# For further details see:
# - github.com/apache/tvm-rfcs/blob/main/rfcs/0060_UMA_Unified_Modular_Accelerator_Interface.md
# - $TVM_HOME/python/tvm/relay/op/strategy/x86.py
