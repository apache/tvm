import tvm
from tvm import te, auto_scheduler

from test_auto_scheduler_common import (matmul_auto_scheduler_test, conv2d_nchw_bn_relu_auto_scheduler_test,
                                        max_pool2d_auto_scheduler_test, min_nm_auto_scheduler_test,
                                        softmax_nm_auto_scheduler_test, softmax_abcd_auto_scheduler_test,
                                        conv2d_winograd_nhwc_auto_scheduler_test)

def print_sketches(sketches):
    for i, s in enumerate(sketches):
        print("=" * 20 + " %d " % i + "=" * 20)
        print(s)

def generate_sketches(workload_func, args, target):
    workload_key = auto_scheduler.make_workload_key(workload_func, args)
    dag = auto_scheduler.ComputeDAG(workload_key)
    task = auto_scheduler.SearchTask(dag, workload_key, tvm.target.create(target))
    policy = auto_scheduler.SketchSearchPolicy(task, verbose=0)
    return policy.generate_sketches()

def test_cpu_matmul_sketch():
    sketches = generate_sketches(matmul_auto_scheduler_test, (512, 512, 512), 'llvm')
    assert len(sketches) == 3
    ''' 3 multi-level tiling sketches
        0 - Multi-level tiling
        1 - Multi-level tiling with cache write on position 0
        2 - Multi-level tiling with cache write on position 1
    '''

    sketches = generate_sketches(matmul_auto_scheduler_test, (8, 8, 512), 'llvm')
    assert len(sketches) == 5
    ''' 2 rfactor sketches + 3 multi-level tiling sketches
        0 - Rfactor with factor position 0
        1 - Rfactor with factor position 1
        2 - Multi-level tiling
        3 - Multi-level tiling with cache write on position 0
        4 - Multi-level tiling with cache write on position 1
    '''

def test_cpu_conv2d_bn_relu_sketch():
    sketches = generate_sketches(conv2d_nchw_bn_relu_auto_scheduler_test,
                                 (1, 56, 56, 512, 512, 3, 1, 1), 'llvm')
    assert len(sketches) == 3
    ''' 3 multi-level tiling sketches
        0 - Conv2d multi-level tiling with fusion on position 0
        1 - Conv2d multi-level tiling with fusion on position 1
        2 - Conv2d multi-level tiling without fusion
    '''

def test_cpu_max_pool2d_sketch():
    sketches = generate_sketches(max_pool2d_auto_scheduler_test, (1, 56, 56, 512, 1), 'llvm')
    assert len(sketches) == 1
    ''' 1 default sketch
    '''

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
    assert len(sketches) == 9
    ''' (2 rfactor sketches + 1 default sketch) * (2 rfactor sketches + 1 default sketch)
    '''

    sketches = generate_sketches(softmax_abcd_auto_scheduler_test, (1, 12, 128, 128), 'llvm')
    assert len(sketches) == 9
    ''' (2 rfactor sketches + 1 default sketch) * (2 rfactor sketches + 1 default sketch)
    '''

def test_cpu_conv2d_winograd_sketch():
    sketches = generate_sketches(conv2d_winograd_nhwc_auto_scheduler_test,
                                 (1, 28, 28, 128, 128, 3, 1, 1), 'llvm')
    assert len(sketches) == 3
    ''' 3 multi-level tiling sketches
        0 - Bgemm multi-level tiling
        1 - Bgemm multi-level tiling with cache write on position 0
        2 - Bgemm multi-level tiling with cache write on position 1
    '''

if __name__ == "__main__":
    test_cpu_matmul_sketch()
    test_cpu_conv2d_bn_relu_sketch()
    test_cpu_max_pool2d_sketch()
    test_cpu_min_sketch()
    test_cpu_softmax_sketch()
    test_cpu_conv2d_winograd_sketch()
