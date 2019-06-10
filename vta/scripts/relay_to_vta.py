"""Perform inference on VTA using Relay."""

import argparse, json, os, requests, time
from io import BytesIO
from mxnet.gluon.model_zoo import vision
import numpy as np
from os.path import join, isfile
from PIL import Image

import tvm
from tvm import rpc, autotvm, relay
from tvm.contrib import graph_runtime, util, download
from tvm.contrib.debugger import debug_runtime
import vta
from vta.testing import simulator
from vta.top import graph_pack


def classification_demo(opt):
    """Image classification demo.

    Parameters
    ----------
    opt: a dictionary obtained from argparse
    """
    
    # Make sure that TVM was compiled with RPC=1
    assert tvm.module.enabled("rpc")

    # Read in VTA environment
    env = vta.get_env()

    # Download ImageNet Categories
    url = "https://github.com/uwsaml/web-data/raw/master/vta/models/"
    categ_fn = "synset.txt"
    for fn in ["synset.txt"]:
        if not isfile(fn):
            download.download(join(url, fn), fn)
    synset = eval(open(categ_fn).read())

    # Download test image
    image_url = 'https://homes.cs.washington.edu/~moreau/media/vta/cat.jpg'
    response = requests.get(image_url)

    # Prepare test image for inference
    image = Image.open(BytesIO(response.content)).resize((224, 224))
    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    image = np.repeat(image, env.BATCH, axis=0)

    # For tuning, make sure tracker variables are set
    tracker_host = os.environ.get("TVM_TRACKER_HOST", None)
    tracker_port = int(os.environ.get("TVM_TRACKER_PORT", None))
    if not tracker_host or not tracker_port:
        print("Set your AutoTVM tracker node host and port variables to run the autotuner")
        exit()

    # We configure both the bitstream and the runtime system on the Pynq
    # to match the VTA configuration specified by the vta_config.json file.
    if env.TARGET != "sim":

        # Measure build start time
        reconfig_start = time.time()

        # Get remote from fleet node
        remote = autotvm.measure.request_remote(env.TARGET, tracker_host, tracker_port, timeout=10000)

        # Reconfigure the JIT runtime and FPGA.
        # You can program the FPGA with your own custom bitstream
        # by passing the path to the bitstream file instead of None.
        vta.reconfig_runtime(remote)
        vta.program_fpga(remote, bitstream=None)

        # Report on reconfiguration time
        reconfig_time = time.time() - reconfig_start
        print("Reconfigured FPGA and RPC runtime in {0:.2f}s!".format(reconfig_time))

    # In simulation mode, host the RPC server locally.
    else:
        remote = rpc.LocalSession()

    # Create a TVM target and execution context
    target = env.target if opt.device == "vta" else env.target_vta_cpu
    ctx = remote.ext_dev(0) if opt.device == "vta" else remote.cpu(0)

    # Get tophub schedules
    with autotvm.tophub.context(target):

        # Measure build start time
        build_start = time.time()

        # Derive the LLVM compiler flags
        # When targetting the Pynq/Ultra-96, cross-compile to ARM ISA
        target_host = env.target_host

        # Populate the shape and data type dictionary
        dtype_dict = {"data": 'float32'}
        shape_dict = {"data": (env.BATCH, 3, 224, 224)}

        # Get off the shelf gluon model, and convert to relay
        gluon_model = vision.get_model(opt.model, pretrained=True)
        relay_prog, params = relay.frontend.from_mxnet(gluon_model, shape_dict)

        # Update shape and type dictionary
        shape_dict.update({k: v.shape for k, v in params.items()})
        dtype_dict.update({k: str(v.dtype) for k, v in params.items()})

        # Perform quantization in Relay
        with relay.quantize.qconfig(global_scale=8.0, skip_k_conv=1):
            relay_prog = relay.quantize.quantize(relay_prog, params=params)

        # Perform graph packing and constant folding for VTA target
        if target.device_name == "vta":
            assert env.BLOCK_IN == env.BLOCK_OUT
            relay_prog = graph_pack(
                relay_prog,
                env.BATCH,
                env.BLOCK_OUT,
                env.WGT_WIDTH,
                start_name=opt.start_name,
                stop_name=opt.stop_name)
            relay_prog = relay.ir_pass.fold_constant(relay_prog)

        # Compile Relay program with AlterOpLayout disabled
        with relay.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
            if target.device_name != "vta":
                graph, lib, params = relay.build(
                    relay_prog, target=target,
                    params=params, target_host=target_host)
            else:
                with vta.build_config():
                    graph, lib, params = relay.build(
                        relay_prog, target=target,
                        params=params, target_host=target_host)
        
        # Measure Relay build time
        build_time = time.time() - build_start
        print(opt.model + " inference graph built in {0:.2f}s!".format(build_time))

        # Send the inference library over to the remote RPC server
        temp = util.tempdir()
        lib.save(temp.relpath("graphlib.o"))
        remote.upload(temp.relpath("graphlib.o"))
        lib = remote.load_module("graphlib.o")

        # If detailed runtime info is needed build with debug runtime
        if opt.debug_profile:
            m = debug_runtime.create(graph, lib, ctx)
        else:
            m = graph_runtime.create(graph, lib, ctx)

        # Set the network parameters and inputs
        m.set_input(**params)
        m.set_input('data', image)

        # Perform inference
        timer = m.module.time_evaluator("run", ctx, number=1, repeat=opt.measurements)
        tcost = timer()

        # Display profile information
        if opt.debug_profile:
            m.run()

        # Get classification results
        tvm_output = m.get_output(0, tvm.nd.empty((env.BATCH, 1000), "float32", remote.cpu(0)))
        top_categories = np.argsort(tvm_output.asnumpy()[0])

        # Report top-5 classification results
        std = np.std(tcost.results) * 1000 / env.BATCH
        mean = tcost.mean * 1000 / env.BATCH
        print("%s Prediction" % opt.model)
        print("                     #1:", synset[top_categories[-1]])
        print("                     #2:", synset[top_categories[-2]])
        print("                     #3:", synset[top_categories[-3]])
        print("                     #4:", synset[top_categories[-4]])
        print("                     #5:", synset[top_categories[-5]])
        print("Performed inference in %.2fms/sample (std = %.2f)" % (mean, std))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--model', type=str, default='resnet18_v1', choices=['resnet18_v1'],
                        help='Input model name.')
    parser.add_argument('--start-name', type=str, default='nn.max_pool2d',
                        help='The name of the node where packing starts')
    parser.add_argument('--stop-name', type=str, default='nn.global_avg_pool2d',
                        help='The name of the node where packing stops')
    parser.add_argument('--debug-profile', action='store_true',
                        help='Show layer-wise time cost profiling results')
    parser.add_argument('--device', default='vta',  choices=['vta', 'arm_cpu'],
                        help='Select device target')
    parser.add_argument('--measurements', type=int, default=1,
                        help='Number of measurements')

    opt = parser.parse_args()

    classification_demo(opt)
