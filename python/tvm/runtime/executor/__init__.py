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

"""This module contains Python wrappers for the TVM C++ Executor implementations.

NOTE: at present, only AOT Executor is contained here. The others are:
 - GraphExecutor, in python/tvm/contrib/graph_executor.py
 - VM Executor, in python/tvm/runtime/vm.py

TODO(areusch): Consolidate these into this module.
"""
from .aot_executor import AotModule
