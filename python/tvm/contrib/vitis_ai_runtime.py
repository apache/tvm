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

"""VitisAI runtime that load and run Xgraph."""
import tvm._ffi

def create(name, model_dir, target):
    """Create a runtime executor module given a xgraph model and context.
    Parameters
    ----------
    model_dir : str
        The directory where the compiled models are located.
    target : str
        The target for running subgraph.

    Returns
    -------
    vai_runtime : VaiModule
        Runtime Vai module that can be used to execute xgraph model.
    """
    runtime_func = "tvm.vitis_ai_runtime.create"
    fcreate = tvm._ffi.get_global_func(runtime_func)
    return VitisAIModule(fcreate(name, model_dir, target))

class VitisAIModule(object):
    """Wrapper runtime module.

    This is a thin wrapper of the underlying TVM module.
    you can also directly call set_input, run, and get_output
    of underlying module functions

    Parameters
    ----------
    module : Module
        The internal tvm module that holds the actual vai functions.

    """

    def __init__(self, module):
        self.module = module
