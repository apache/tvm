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
"""Common utilities for testing autotvm"""
import time

import tvm
from tvm import autotvm
from tvm.autotvm import MeasureInput, MeasureResult

@autotvm.template
def matmul(N, L, M, dtype):
    A = tvm.placeholder((N, L), name='A', dtype=dtype)
    B = tvm.placeholder((L, M), name='B', dtype=dtype)

    k = tvm.reduce_axis((0, L), name='k')
    C = tvm.compute((N, M), lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=k), name='C')
    s = tvm.create_schedule(C.op)

    # schedule
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    ##### define space begin #####
    cfg = autotvm.get_config()
    cfg.define_split("tile_y", y, num_outputs=2)
    cfg.define_split("tile_x", x, num_outputs=2)
    ##### define space end #####

    # schedule according to config
    yo, yi = cfg["tile_y"].apply(s, C, y)
    xo, xi = cfg["tile_x"].apply(s, C, x)

    s[C].reorder(yo, xo, k, yi, xi)

    return s, [A, B, C]

@autotvm.template
def bad_matmul(N, L, M, dtype):
    if 'bad_device' in tvm.target.current_target().keys:
        A = tvm.placeholder((N, L), name='A', dtype=dtype)
        B = tvm.placeholder((L, M), name='B', dtype=dtype)

        k = tvm.reduce_axis((0, L-1), name='k')
        C = tvm.compute((N, M), lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=k), name='C')
        s = tvm.create_schedule(C.op)

        # schedule
        y, x = s[C].op.axis
        cfg = autotvm.get_config()
        cfg.define_split("tile_y", y, num_outputs=2)
        cfg.define_split("tile_x", x, num_outputs=2)
        return s, [A, B, C]

    return matmul(N, L, M, dtype)

def get_sample_task(n=128):
    """return a sample task for testing"""
    target = tvm.target.create("llvm")
    task = autotvm.task.create(matmul, args=(n, n, n, 'float32'), target=target)
    return task, target

def get_sample_records(n):
    """get sample records for testing"""
    tsk, target = get_sample_task()

    inps, ress = [], []
    for i in range(n):
        inps.append(MeasureInput(target, tsk, tsk.config_space.get(i)))
        ress.append(MeasureResult((i+1,), 0, i, time.time()))
    return list(zip(inps, ress))

