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
Using the template-free auto-scheduler on CPU 
=============================================
**Author**: `Lianmin Zheng <https://github.com/merrymercy>`_, \
            `Chengfan Jia <https://github.com/jcf94/>`_

This is a tutorial on how to use the auto-scheduler in TVM.

Different from the exiting autotvm which relies on manual templates to 
define the search space, the auto-scheduler does not require any templates.
The user only needs to write the computation declaration,
the auto-scheduler then automatically generate a large
search space and begins the search (or auto-tuning).

We use matrix multiplication as an example in this tutorial.
"""

import numpy as np
import tvm
from tvm import te, testing, auto_scheduler

######################################################################
# To begin with, we define the computation of a matmul with bias add.
# The function should return the list of input/output tensors.
# From these tensors, the auto-scheduler can get the whole computational graph.

@auto_scheduler.register_workload
def matmul_add(N, L, M, dtype):
    A = te.placeholder((N, L), name='A', dtype=dtype)
    B = te.placeholder((L, M), name='B', dtype=dtype)
    C = te.placeholder((N, M), name='C', dtype=dtype)

    k = te.reduce_axis((0, L), name='k')
    matmul = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
                        name='matmul')
    D = te.compute((N, M), lambda i, j: matmul[i, j] + C[i, j], name='D')

    return [A, B, C, D]

######################################################################
# We then create the a search task with N=L=M=128 and dtype='float32'

target = tvm.target.Target("llvm")
task = auto_scheduler.create_task(matmul_add, (128, 128, 128, 'float32'), target)

print(task.compute_dag)

######################################################################
# Next, we set parameters for the auto-scheduler.
# `num_measure_trials` is the number of measurement trials we can use during the search.
# We only make 10 trials in this tutorial for fast demonstration. In practice, 1000 is a good value for
# the search to converge. You can do more trials according to your time budget.
# In addition, we use `RecordToFile` to log measurement records into a file `test.json`.
# The measurement records can be used to query the history best, resume the search,
# or train the cost model later.

tune_option = auto_scheduler.TuningOptions(num_measure_trials=2,
                                           measure_callbacks=[auto_scheduler.RecordToFile('test.json')])

######################################################################
# Now we get all inputs ready. Pretty simple, isn't it?
# We can kick off the search and let the auto-scheduler do its magic.
# After some measurement trials, it will return the best schedule it founds.

sch, args = auto_scheduler.auto_schedule(task,
                                         tuning_options=tune_option)

######################################################################
# We can lower schedule to see the IR after auto-scheduling.
# We can also build the binary function as usual.

print(tvm.lower(sch, args, simple_mode=True))
func = tvm.build(sch, args)

######################################################################
# Finally, let use do a correctness check

# check correctness
a_np = np.random.uniform(size=(128, 128)).astype(np.float32)
b_np = np.random.uniform(size=(128, 128)).astype(np.float32)
c_np = np.random.uniform(size=(128, 128)).astype(np.float32)
d_np = a_np.dot(b_np) + c_np

d_tvm = tvm.nd.empty(d_np.shape)
func(tvm.nd.array(a_np), tvm.nd.array(b_np), tvm.nd.array(c_np), d_tvm)

tvm.testing.assert_allclose(d_np, d_tvm.asnumpy(), rtol=1e-2)