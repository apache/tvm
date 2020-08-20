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
""" Test evolutionary search. """

import tvm
from tvm import te, auto_scheduler

from test_auto_scheduler_common import conv2d_nchw_bn_relu_auto_scheduler_test

def test_evo_search():
    workload_key = auto_scheduler.make_workload_key(conv2d_nchw_bn_relu_auto_scheduler_test,
                                                    (1, 56, 56, 512, 512, 3, 1, 1))
    dag = auto_scheduler.ComputeDAG(workload_key)
    task = auto_scheduler.SearchTask(dag, workload_key, tvm.target.create('llvm'))
    policy = auto_scheduler.SketchPolicy(task, verbose=0)
    states = policy.sample_initial_population(50)
    policy.evolutionary_search(states, 10)


if __name__ == "__main__":
    test_evo_search()