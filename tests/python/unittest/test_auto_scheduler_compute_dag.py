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

"""Test ComputeDAG (replay, infer bound)"""

import tvm
from tvm import auto_scheduler, te

from test_auto_scheduler_common import get_tiled_matmul


def test_apply_steps():
    dag, s = get_tiled_matmul()
    dag.print_python_code_from_state(s)
    sch, tensors = dag.apply_steps_from_state(s)
    stmt = tvm.lower(sch, tensors, simple_mode=True)


def test_infer_bound():
    dag, s = get_tiled_matmul()
    s = dag.infer_bound_from_state(s)


def test_estimate_flop():
    dag, s = get_tiled_matmul()
    assert abs(dag.flop_ct - 2 * 512 ** 3) < 0.5


if __name__ == "__main__":
    test_apply_steps()
    test_infer_bound()
    test_estimate_flop()
