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
# pylint: disable=consider-using-with, unnecessary-ellipsis

"""This module is to test the subgraph module that is in /python/tvm/contrib/hexagon/"""
import sys
import logging
import tvm
from tvm.contrib.hexagon.subgraph import smallest_ir
from tvm import relay
from tvm.relay.backend import Executor
from tvm.contrib.hexagon.pytest_plugin import HEXAGON_AOT_LLVM_TARGET
#logger = logging.getLogger()
#logger.setLevel(logging.DEBUG)

if len(sys.argv) != 2:
    raise RuntimeError("You should give a target as an argument to the file")

#choosing the target
if sys.argv[1] == "hexagon":
    target = HEXAGON_AOT_LLVM_TARGET
elif sys.argv[1] == "x86":
    target = tvm.target.Target("llvm")
else:
    raise RuntimeError(f'The {sys.argv[1]} target is not supported')


inp = relay.var("data", shape=(1,32,32,128), dtype="float32")
l1_weights = relay.var("l1_weights", shape=(3,3,128,256), dtype="float32")
l1_bias = relay.var("bias", shape=(256,), dtype="float32")
l1_conv = relay.nn.conv2d(inp, l1_weights, kernel_size = [3, 3], data_layout = "NHWC",
kernel_layout = "HWIO", out_dtype="float32", padding=[1,1], strides=(2,2))
l1_bias_add = relay.nn.bias_add(l1_conv, l1_bias, axis=3)
l1_relu = relay.nn.relu(l1_bias_add)
l2_weights = relay.var("l2_weights", shape=(3,3,256,512), dtype="float32")
l2_bias = relay.var("bias2", shape=(512,), dtype="float32")

def is_compiled(mod, targ):
    """Function to check whether the current module is able to be compiled or not"""
    try:
        params = {}
        executor = Executor("aot", {"link-params": True})
        lowered = tvm.relay.build(
            mod,
            tvm.target.Target(targ, host=targ),
            executor=executor,
            params=params,
        )
        logging.debug(lowered)
        return True
    except: # pylint: disable=W0702
        return False

def test_case(boo):
    """creating test case"""
    if boo:
        l2_conv = relay.nn.conv2d(l1_relu, l2_weights, kernel_size = [3, 3], data_layout = "CWHN",
	kernel_layout = "HWIO", out_dtype="float32", padding=[1,1], strides=(2,2))
    else:
        l2_conv = relay.nn.conv2d(l1_relu, l2_weights, kernel_size = [3, 3], data_layout = "NHWC",
	kernel_layout = "HWIO", out_dtype="float32", padding=[1,1], strides=(2,2))
    l2_bias_add = relay.nn.bias_add(l2_conv, l2_bias, axis=3)
    l2_relu = relay.nn.relu(l2_bias_add)

    pooling = relay.nn.avg_pool2d(l2_relu, pool_size=(8,8), layout="NHWC")
    reshape = relay.reshape(pooling, newshape=[1,512])

    dense_weights = relay.var("dense_weights", shape=(10,512), dtype="float32")
    dense = relay.nn.dense(reshape, dense_weights)
    softmax = relay.nn.softmax(dense, axis=1)
    mod = tvm.IRModule.from_expr(softmax)
    return mod


#manually creating expected output
result = relay.nn.conv2d(relay.var("temp", type_annotation=relay.transform.InferTypeLocal(l1_relu)),
	          relay.var("temp1", type_annotation=relay.transform.InferTypeLocal(l2_weights)),
                  kernel_size = [3, 3], data_layout = "CWHN", kernel_layout = "HWIO",
                  out_dtype="float32", padding=[1,1], strides=(2,2))
print("module read")
func = relay.Function(relay.analysis.free_vars(relay.expr.Tuple([result])),
                      relay.expr.Tuple([result]))
mod_result = tvm.IRModule.from_expr(func)

#print(smallest_ir(test_case(True), target, is_compiled))
#checking if the result is as expected
print(tvm.ir.structural_equal(mod_result, smallest_ir(test_case(True), target, is_compiled)))
print(tvm.ir.structural_equal(None, smallest_ir(test_case(False), target, is_compiled)))
#print(smallest_ir(mod, target))
#import pdb; pdb.set_trace()
