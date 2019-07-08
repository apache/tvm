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
import numpy as np
from tvm import relay
from tvm.relay.testing import create_workload
from tvm.contrib import graph_runtime

# TODOs for janimesh before submitting this patch.
# TODO - Add tests for int8 input/weight dtype
# TODO - opt_level=0 fails mostly due to fusion.
# TODO - opt_level=3 fails, likely culprit kernel layout for int8
# compute. Work with Rankyung to see if this is the culprit. Handle
# it in a separate patch.

def run_infer_type(expr):
    mod = relay.Module.from_expr(expr)
    mod = relay.transform.InferType()(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body



if __name__ == "__main__":
    # add your tests here.
    pass
