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

""" Test sketch generation. """

import tvm
from tvm import te, auto_scheduler

from test_auto_scheduler_common import (matmul_auto_scheduler_test, conv2d_nchw_bn_relu_auto_scheduler_test,
                                        max_pool2d_auto_scheduler_test, min_nm_auto_scheduler_test,
                                        softmax_nm_auto_scheduler_test, softmax_abcd_auto_scheduler_test,
                                        conv2d_winograd_nhwc_auto_scheduler_test)

def generate_sketches(workload_func, args, target, print_for_debug=False):
    workload_key = auto_scheduler.make_workload_key(workload_func, args)
    dag = auto_scheduler.ComputeDAG(workload_key)
    task = auto_scheduler.SearchTask(dag, workload_key, tvm.target.create(target))
    policy = auto_scheduler.SketchPolicy(task, verbose=0)
    return policy.generate_sketches(print_for_debug)

def test_cpu_matmul_sketch():
    sketches = generate_sketches(matmul_auto_scheduler_test, (512, 512, 512), 'llvm')
    ''' 3 multi-level tiling sketches
        0 - Multi-level tiling
        1 - Multi-level tiling with cache write on position 0
        2 - Multi-level tiling with cache write on position 1
    '''
    assert len(sketches) == 3

    sketches = generate_sketches(matmul_auto_scheduler_test, (8, 8, 512), 'llvm')
    ''' 2 rfactor sketches + 3 multi-level tiling sketches
        0 - Rfactor with factor position 0
        1 - Rfactor with factor position 1
        2 - Multi-level tiling
        3 - Multi-level tiling with cache write on position 0
        4 - Multi-level tiling with cache write on position 1
    '''
    assert len(sketches) == 5

def test_cpu_conv2d_bn_relu_sketch():
    sketches = generate_sketches(conv2d_nchw_bn_relu_auto_scheduler_test,
                                 (1, 56, 56, 512, 512, 3, 1, 1), 'llvm')
    ''' 3 multi-level tiling sketches
        0 - Conv2d multi-level tiling with fusion on position 0
        1 - Conv2d multi-level tiling with fusion on position 1
        2 - Conv2d multi-level tiling without fusion
    '''
    assert len(sketches) == 3

def test_cpu_max_pool2d_sketch():
    sketches = generate_sketches(max_pool2d_auto_scheduler_test, (1, 56, 56, 512, 1), 'llvm')
    assert len(sketches) == 1  # 1 default sketch

def test_cpu_min_sketch():
    sketches = generate_sketches(min_nm_auto_scheduler_test, (10, 1024), 'llvm')
    assert len(sketches) == 3
    ''' 2 rfactor sketches + 1 default sketch
        0 - Rfactor with factor position 0
        1 - Rfactor with factor position 1
        2 - Default sketch
    '''

def test_cpu_softmax_sketch():
    sketches = generate_sketches(softmax_nm_auto_scheduler_test, (1, 1024), 'llvm')
    ''' (2 rfactor sketches + 1 default sketch) * (2 rfactor sketches + 1 default sketch) '''
    assert len(sketches) == (3 * 3)

    sketches = generate_sketches(softmax_abcd_auto_scheduler_test, (1, 12, 128, 128), 'llvm')
    ''' (2 rfactor sketches + 1 default sketch) * (2 rfactor sketches + 1 default sketch) '''
    assert len(sketches) == (3 * 3)

def test_cpu_conv2d_winograd_sketch():
    sketches = generate_sketches(conv2d_winograd_nhwc_auto_scheduler_test,
                                 (1, 28, 28, 128, 128, 3, 1, 1), 'llvm')
    ''' 3 multi-level tiling sketches
        0 - Bgemm multi-level tiling
        1 - Bgemm multi-level tiling with cache write on position 0
        2 - Bgemm multi-level tiling with cache write on position 1
    '''
    assert len(sketches) == 3

if __name__ == "__main__":
    test_cpu_matmul_sketch()
    test_cpu_conv2d_bn_relu_sketch()
    test_cpu_max_pool2d_sketch()
    test_cpu_min_sketch()
    test_cpu_softmax_sketch()
    test_cpu_conv2d_winograd_sketch()
