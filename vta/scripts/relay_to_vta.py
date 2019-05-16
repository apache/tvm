"""Perform inference on VTA using Relay."""

import argparse, json, requests, time
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

parser = argparse.ArgumentParser(description='Train a model for image classification.')
parser.add_argument('--model', type=str, required=True,
                    help='Input model name.')
parser.add_argument('--start-name', type=str, default='nn.max_pool2d',
                    help='The name of the node where packing starts')
parser.add_argument('--stop-name', type=str, default='nn.global_avg_pool2d',
                    help='The name of the node where packing stops')
parser.add_argument('--debug-profile', action='store_true',
                    help='Show layer-wise time cost profiling results')
parser.add_argument('--device', default="vta",
                    help='Select device target, either "vta" or "vtacpu"')
parser.add_argument('--measurements', type=int, default=1,
                    help='Number of measurements')

opt = parser.parse_args()

if 'mobilenet' in opt.model:
    opt.start_name = 'nn.relu'
elif 'gan' in opt.model:
    opt.start_name = 'reshape0'
    opt.stop_name = 'copy2'
elif 'rnn' in opt.model:
    opt.start_name = 'reshape0'
    opt.stop_name = 'reshape1'

# Helper function to read in image
# Takes in Image object, returns an ND array
def process_image(image):
    # Convert to neural network input format
    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]

    return tvm.nd.array(image.astype("float32"))

def demo_cat_classification(env, m, ctx, remote, shape_dict, dtype_dict):
    # Read in ImageNet Categories
    url = "https://github.com/uwsaml/web-data/raw/master/vta/models/"
    categ_fn = "synset.txt"
    for fn in ["synset.txt"]:
        if not isfile(fn):
            download.download(join(url, fn), fn)
    synset = eval(open(categ_fn).read())
    # Read in test image
    image_url = 'https://homes.cs.washington.edu/~moreau/media/vta/cat.jpg'
    # Read in test image
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).resize((224, 224))
    # Set the input
    image = process_image(image)
    if "gan" in opt.model or "rnn" in opt.model:
        # non-classification networks require custom input shapes and out shapes
        m.set_input('data', tvm.nd.array(
            10 * np.random.uniform(size=shape_dict['data']).astype(dtype_dict['data'])))
        timer = m.module.time_evaluator("run", ctx, number=1, repeat=opt.measurements)
        tcost = timer()
        std = np.std(tcost.results) * 1000 / env.BATCH
        mean = tcost.mean * 1000 / env.BATCH
        print("Performed inference in %.2fms/samlple (std = %.2f)" % (mean, std))
    else:
        image = np.repeat(image.asnumpy(), env.BATCH, axis=0)
        m.set_input('data', image)
        # Perform inference
        timer = m.module.time_evaluator("run", ctx, number=1, repeat=opt.measurements)
        tcost = timer()

        if opt.debug_profile:
            m.run()

        # Get classification results
        tvm_output = m.get_output(0,
                                  tvm.nd.empty((env.BATCH, 1000), "float32", remote.cpu(0)))
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

######################################################################
# Setup the Pynq Board's RPC Server
# ---------------------------------
# Build the RPC server's VTA runtime and program the Pynq FPGA.

def run(device = "vta"):
    env = vta.get_env()
    # Measure build start time
    reconfig_start = time.time()

    # We configure both the bitstream and the runtime system on the Pynq
    # to match the VTA configuration specified by the vta_config.json file.
    if env.TARGET != "sim":

        # Make sure that TVM was compiled with RPC=1
        assert tvm.module.enabled("rpc")

        # Get remote from fleet node
        remote = autotvm.measure.request_remote(env.TARGET, '10.77.1.109', 9190, timeout=10000)

        # Reconfigure the JIT runtime
        vta.reconfig_runtime(remote)

        # Program the FPGA with a pre-compiled VTA bitstream.
        # You can program the FPGA with your own custom bitstream
        # by passing the path to the bitstream file instead of None.
        vta.program_fpga(remote, bitstream=None)

        # Report on reconfiguration time
        reconfig_time = time.time() - reconfig_start
        print("Reconfigured FPGA and RPC runtime in {0:.2f}s!".format(reconfig_time))

    # In simulation mode, host the RPC server locally.
    elif env.TARGET == "sim":
        remote = rpc.LocalSession()

    # TVM target and context
    target = tvm.target.create("llvm -device={}".format(device))
    ctx = remote.ext_dev(0) if device == "vta" else remote.cpu(0)

    with autotvm.tophub.context(target):

        # Measure build start time
        build_start = time.time()

        # Derive the LLVM compiler flags
        # When targetting the Pynq/Ultra-96, cross-compile to ARM ISA
        target_host = env.target_host

        # Populate the shape and data type dictionary
        dtype_dict = {"data": 'float32'}
        if "gan" in opt.model:
            shape_dict = {"data": (env.BATCH, 100)}
        elif 'rnn' in opt.model:
            batch_size, seq_len, hidden_dim = 4, 1, 640
            begin_state_shape = (batch_size, hidden_dim, 1, 1)
            shape_dict = {"data": (seq_len, batch_size),
                        "cell_l0_begin_state_0": begin_state_shape,
                        "cell_l1_begin_state_0": begin_state_shape}
            dtype_dict = {"data": "int32",
                        "cell_l0_begin_state_0": 'float32',
                        "cell_l1_begin_state_0": 'float32'}
        else:
            shape_dict = {"data": (env.BATCH, 3, 224, 224)}

        gluon_model = vision.get_model(opt.model, pretrained=True)
        relay_graph, params = relay.frontend.from_mxnet(gluon_model, shape_dict)

        shape_dict.update({k: v.shape for k, v in params.items()})
        dtype_dict.update({k: str(v.dtype) for k, v in params.items()})

        with relay.quantize.qconfig(global_scale=8.0, skip_k_conv=1):
            relay_graph = relay.quantize.quantize(relay_graph, params=params)

        if target.device_name == "vta":
            assert env.BLOCK_IN == env.BLOCK_OUT
            relay_graph = graph_pack(
                relay_graph,
                env.BATCH,
                env.BLOCK_OUT,
                env.WGT_WIDTH,
                start_name=opt.start_name,
                stop_name=opt.stop_name)

            relay_graph = relay.ir_pass.fold_constant(relay_graph)

        # Compile Relay program.
        with relay.build_module.build_config(opt_level=3, disable_pass={"AlterOpLayout"}):
            if target.device_name != "vta":
                graph, lib, params = relay.build(
                    relay_graph, target=target,
                    params=params, target_host=target_host)
            else:
                with vta.build_config():
                    graph, lib, params = relay.build(
                        relay_graph, target=target,
                        params=params, target_host=target_host)


        # Save the compiled inference graph library
        assert tvm.module.enabled("rpc")
        temp = util.tempdir()
        lib.save(temp.relpath("graphlib.o"))

        # Send the inference library over to the remote RPC server
        remote.upload(temp.relpath("graphlib.o"))
        lib = remote.load_module("graphlib.o")

        # Measure build time
        build_time = time.time() - build_start
        print(opt.model + " inference graph built in {0:.2f}s!".format(build_time))

        if opt.debug_profile:
            m = debug_runtime.create(graph, lib, ctx)
        else:
            m = graph_runtime.create(graph, lib, ctx)

        # Set the parameters
        m.set_input(**params)
        demo_cat_classification(env, m, ctx, remote, shape_dict, dtype_dict)

run(opt.device)
