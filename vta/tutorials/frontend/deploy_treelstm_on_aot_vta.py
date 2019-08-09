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
from __future__ import absolute_import, print_function

import argparse, json, os, requests, time
from io import BytesIO
from os.path import join, isfile
from PIL import Image

from mxnet.gluon.model_zoo import vision
import numpy as np
from matplotlib import pyplot as plt

import tvm
from tvm import rpc, autotvm, relay
from tvm.contrib import graph_runtime, util, download
from tvm.contrib.debugger import debug_runtime
from tvm.relay.transform import PartialEvaluate, ToANormalForm, DeadCodeElimination, ToGraphNormalForm, FuseOps

import vta
from vta.testing import simulator
from vta.top import graph_pack, GraphPack
import aot
import network
from network.tlstm import TreeLSTM

# Make sure that TVM was compiled with RPC=1
assert tvm.module.enabled("rpc")

env = vta.get_env()

device = "vta"
target = env.target if device == "vta" else env.target_vta_cpu

# The ``start_pack`` and ``stop_pack`` labels indicate where
# to start and end the graph packing relay pass: in other words
# where to start and finish offloading to VTA.
model = "tree_lstm"
start_pack="nn.max_pool2d"
stop_pack="nn.global_avg_pool2d"

######################################################################
# Obtain an execution remote
# ---------------------------------
# When target is 'pynq', reconfigure FPGA and runtime.
# Otherwise, if target is 'sim', execute locally.
if env.TARGET not in ["sim", "tsim"]:

    # Get remote from tracker node if environment variable is set.
    # To set up the tracker, you'll need to follow the "Auto-tuning
    # a convolutional network for VTA" tutorial.
    tracker_host = os.environ.get("TVM_TRACKER_HOST", None)
    tracker_port = os.environ.get("TVM_TRACKER_PORT", None)
    # Otherwise if you have a device you want to program directly from
    # the host, make sure you've set the variables below to the IP of
    # your board.
    device_host = os.environ.get("VTA_PYNQ_RPC_HOST", "192.168.2.99")
    device_port = os.environ.get("VTA_PYNQ_RPC_PORT", "9091")
    if not tracker_host or not tracker_port:
        remote = rpc.connect(device_host, int(device_port))
    else:
        remote = autotvm.measure.request_remote(env.TARGET, tracker_host, int(tracker_port), timeout=10000)

    # Reconfigure the JIT runtime and FPGA.
    # You can program the FPGA with your own custom bitstream
    # by passing the path to the bitstream file instead of None.
    reconfig_start = time.time()
    vta.reconfig_runtime(remote)
    vta.program_fpga(remote, bitstream=None)
    reconfig_time = time.time() - reconfig_start
    print("Reconfigured FPGA and RPC runtime in {0:.2f}s!".format(reconfig_time))

# In simulation mode, host the RPC server locally.
else:
    remote = rpc.LocalSession()

# Get execution context from remote
ctx = remote.ext_dev(0) if device == "vta" else remote.cpu(0)

from tvm.relay import ExprMutator, ExprVisitor



treelstm = TreeLSTM(input_size=128, memory_size=256, dtype="int8")
mod = treelstm.mod
p = treelstm.p
mod = ToANormalForm()(mod)
mod = PartialEvaluate()(mod)
mod = DeadCodeElimination()(mod)
mod["main"] = treelstm.get()

import pprint

print(mod["f_0"])
expr_layouts, ops_result_layouts = layout_vta(mod["f_0"])
rewritten_f0 = rewrite_vta(mod["f_0"], expr_layouts, ops_result_layouts)
mod["f_0"] = rewritten_f0
#mod = ToGraphNormalForm()(mod)
#mod = FuseOps()(mod)
#mod = ToANormalForm()(mod)

# Load pre-configured AutoTVM schedules
with autotvm.tophub.context(target):
    # Compile Relay program with AlterOpLayout disabled
    with relay.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
        assert target.device_name == "vta"
        with vta.build_config():
            f = aot.compile(mod["main"], mod, ctx, target)

import torch
import tvm
from tvm import relay
from tvm.relay.backend.interpreter import Value, TupleValue, ConstructorValue, TensorValue
from tvm.relay import testing, create_executor, Constructor
from tvm.relay.prelude import Prelude

class RoseTree:
    def __init__(self, head, children):
        self.head = head
        self.children = children

    def __str__(self):
        return "Tree(" + str(self.head) + ", " + str(self.children) + ")"

    def __repr__(self):
        return self.__str__()

    def fmap(self, f):
        return RoseTree(f(self.head), [x.fmap(f) for x in self.children])

    def size(self):
        return 1 + sum([x.size() for x in self.children])

def rand_tree(depth=3, branch=3):
    shape = (1, 128)
    head = np.random.normal(0, 100, shape).astype("int8")
    children = [rand_tree(depth-1, branch) for x in range(0 if depth == 0 else branch)]
    return RoseTree(head, children)

# creates relay list from a list
def from_list(p, l):
    if len(l) == 0:
        return (p.nil,)
    else:
        return (p.cons, l[0], from_list(p, l[1:]))

def from_tree(p, rt):
    return (p.rose,
            rt.head,
            from_list(p, [from_tree(p, x) for x in rt.children]))


assert env.TARGET in ["sim", "tsim"]
simulator.clear_stats()

def run():
    tvm_output = f(from_tree(p, rand_tree()))
    sim_stats = simulator.stats()
    print("\nExecution statistics:")
    for k, v in sim_stats.items():
        # Since we execute the workload many times, we need to normalize stats
        # Note that there is always one warm up run
        # Therefore we divide the overall stats by (num * rep + 1)
        print("\t{:<16}: {:>16}".format(k, v))
