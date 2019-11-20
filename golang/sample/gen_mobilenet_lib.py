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

import tvm
from tvm import relay
import tvm.relay.testing

######################################################################
# Load Neural Network in Relay
####################################################

mod, params = relay.testing.mobilenet.get_workload(batch_size=1)

# set show_meta_data=True if you want to show meta data
print(mod.astext(show_meta_data=False))

######################################################################
# Compilation
####################################################

target = 'llvm'

# Build with Relay
with relay.build_config(opt_level=0):
    graph, lib, params = relay.build_module.build(
        mod, target, params=params)

######################################################################
# Save and Load Compiled Module
# -----------------------------
# We can also save the graph, lib and parameters into files
####################################################

lib.export_library("./mobilenet.so")
print('lib export succeefully')

with open("./mobilenet.json", "w") as fo:
   fo.write(graph)

with open("./mobilenet.params", "wb") as fo:
   fo.write(relay.save_param_dict(params))
