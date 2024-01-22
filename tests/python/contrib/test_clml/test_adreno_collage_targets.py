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

"""Compares Collage with various other baselines."""

import tvm
import logging
import os
import numpy as np
from tvm.relay import testing
from tvm.relay.build_module import bind_params_by_name
from tvm import autotvm
from tvm.relay.collage.collage import *
from test_clml.infrastructure import compile_and_run
import pytest

logging.basicConfig(level=logging.INFO)


###
### How aggressively to look for candidates?
###
TVM_MAX_DEPTH = 8
BYOC_MAX_DEPTH = 8

###
### TVM Opencl AutoTvm log file name
###
TUNING_LOG = ""

##
## Default Target definition
##
HOST = tvm.target.Target("llvm -mtriple=arm64-linux-android")
OPENCL = tvm.target.Target("opencl -device=adreno", HOST)
RPC_TRACKER_HOST = os.getenv("TVM_TRACKER_HOST", "localhost")
RPC_TRACKER_PORT = int(os.getenv("TVM_TRACKER_PORT", 9090))
RPC_KEY = os.getenv("RPC_DEVICE_KEY", "android")
NDK_CC = os.getenv("TVM_NDK_CC", "aarch64-linux-android-g++")


def get_rpc_remote():
    """Create remote rpc tracker and connect to available remote device"""
    tracker = rpc.connect_tracker(RPC_TRACKER_HOST, RPC_TRACKER_PORT)
    remote = tracker.request(RPC_KEY, priority=0, session_timeout=600)
    return remote


def collage(model, input_data):
    """Run the Collage partitioner for a set of Opencl Adreno related targets and profile the result"""
    logging.info(f"collage | {model['name']}")
    logging.info("-------------- BEGIN ORIGINAL --------------")
    logging.info(model["mod"])
    logging.info("-------------- END ORIGINAL ----------------")
    with autotvm.apply_history_best(TUNING_LOG):
        targets = []
        targets.append(OPENCL)
        use_fp16 = model["main_dtype"] == "float16"
        targets.append(tvm.target.Target("clml", HOST))

        # Register byoc fusion style for compiler with available
        # options [compiler.NoFusion | compiler.TVMFusion | compiler.MaxDepthFusion]
        config = {
            "relay.collage.tvm_max_depth": TVM_MAX_DEPTH,
            "relay.collage.byoc_max_depth": BYOC_MAX_DEPTH,
            "relay.collage.byoc_fusion_style": ["clml.NoFusion"],
        }
        logging.info(f"Using PassContext(config={config}")
        ctxt = tvm.transform.PassContext(config=config)
        config = tvm.target.make_compilation_config(ctxt, targets)
        with ctxt:
            mod = model["mod"]
            """Collage partition with tvm opencl and clml target on rpc device"""
            mod = tvm.relay.transform.CollagePartition(
                config,
                cost_estimator=CostEstimator(
                    host=RPC_TRACKER_HOST, port=RPC_TRACKER_PORT, rpc_key=RPC_KEY, ndk_cc=NDK_CC
                ),
            )(mod)
            partitioned_model = model.copy()
            partitioned_model["mod"] = mod
            logging.info("-------------- BEGIN PARTITIONED --------------")
            logging.info(partitioned_model["mod"])
            logging.info("-------------- END PARTITIONED ----------------")
            return compile_and_run(
                get_rpc_remote(), "collage", partitioned_model, targets, input_data
            )


def just_clml(model, input_data):
    """Run partition_for_clml, complete the compilation with TVM, and profile the result."""
    logging.info(f"just_clml | {model['name']}")
    logging.info("-------------- BEGIN ORIGINAL --------------")
    logging.info(model["mod"])
    logging.info("-------------- END ORIGINAL ----------------")
    with tvm.transform.PassContext(opt_level=3):
        logging.info("Partitioning for CLML...")
        mod = tvm.relay.op.contrib.clml.partition_for_clml(model["mod"], model["params"])
        partitioned_model = model.copy()
        partitioned_model["mod"] = mod
        logging.info("-------------- BEGIN PARTITIONED --------------")
        logging.info(partitioned_model["mod"])
        logging.info("-------------- END PARTITIONED ----------------")
        targets = []
        targets.append(OPENCL)
        targets.append(tvm.target.Target("clml", HOST))
        return compile_and_run(get_rpc_remote(), "just_clml", partitioned_model, OPENCL, input_data)


def just_tvm(model, input_data):
    """Compile and profile using vanilla TVM."""
    logging.info(f"just_tvm | {model['name']}")
    logging.info("-------------- BEGIN ORIGINAL --------------")
    logging.info(model["mod"])
    logging.info("-------------- END ORIGINAL ----------------")
    with autotvm.apply_history_best(TUNING_LOG):
        with tvm.transform.PassContext(opt_level=3):
            return compile_and_run(get_rpc_remote(), "just_tvm", model, OPENCL, input_data)


def get_model(model_name, dtype):

    if "mobilenet" in model_name:
        mod, params = testing.mobilenet.get_workload(batch_size=1, dtype=dtype)
    elif "resnet" in model_name:
        mod, params = testing.resnet.get_workload(num_layers=50, batch_size=1, dtype=dtype)

    initializer = tvm.relay.testing.init.Xavier()
    for param_name in list(params.keys()):
        filter_data = np.zeros(params[param_name].shape).astype(params[param_name].dtype)
        if len(filter_data.shape) > 1:
            initializer("weight", filter_data)
        else:
            initializer("bias", filter_data)
        params[param_name] = tvm.nd.array(filter_data)

    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)
        mod = tvm.relay.transform.FoldConstant()(mod)
    return {
        "name": model_name,
        "input_shapes": {"data": [1, 3, 224, 224]},
        "input_dtypes": {"data": dtype},
        "mod": mod,
        "params": params,
        "main_dtype": dtype,
    }


########### Runners ###########
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("model_name", ["mobilenet", "resnet-50"])
@tvm.testing.requires_openclml
def test_network_collage(model_name, dtype):
    print("Network evaluating .. " + model_name + " " + dtype)
    np.random.seed(0)
    model = get_model(model_name, dtype)
    input_data = {}
    for name, shape in model["input_shapes"].items():
        input_data[name] = np.random.uniform(-1.0, 1.0, shape).astype(model["input_dtypes"][name])

    clml_out = just_clml(model, input_data)
    tvm_out = just_tvm(model, input_data)
    """Check tvm and clml output correctness."""
    tvm_sort = np.argsort(tvm_out).flatten()
    clml_sort = np.argsort(clml_out).flatten()
    tvm.testing.assert_allclose(tvm_sort[-5:], clml_sort[-5:], rtol=0, atol=0)
    logging.info("-------- TVM and CLML execution test passed ---------")

    """Run Collage for tvm and clml compiler target."""
    collage_out = collage(model, input_data)
    collage_sort = np.argsort(collage_out).flatten()
    """Check tvm and collage(tvm+clml) output correctness."""
    tvm.testing.assert_allclose(tvm_sort[-5:], collage_sort[-5:], rtol=0, atol=0)
    logging.info("-------- Collage execution test passed ---------")


if __name__ == "__main__":
    tvm.testing.main()
