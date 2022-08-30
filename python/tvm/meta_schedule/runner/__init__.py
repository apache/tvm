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
"""
The tvm.meta_schedule.runner package.
Meta Schedule runners that runs an artifact either locally or through the RPC interface
"""
from .config import EvaluatorConfig, RPCConfig
from .rpc_runner import RPCRunner
from .local_runner import LocalRunner, LocalRunnerFuture
from .runner import PyRunner, Runner, RunnerFuture, RunnerInput, RunnerResult, PyRunnerFuture
