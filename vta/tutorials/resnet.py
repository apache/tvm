"""
ResNet Inference Example
========================
**Author**: `Thierry Moreau <https://homes.cs.washington.edu/~moreau/>`_

This tutorial provides an end-to-end demo, on how to run ResNet-18 inference
onto the VTA accelerator design to perform ImageNet classification tasks.

"""

######################################################################
# Import Libraries
# ----------------
# We start by importing the tvm, vta, nnvm libraries to run this example.

from __future__ import absolute_import, print_function

import os
import time
from io import BytesIO

import numpy as np
import requests
from matplotlib import pyplot as plt
from PIL import Image

import tvm
from tvm import rpc, autotvm
from tvm.contrib import graph_runtime, util
from tvm.contrib.download import download
import nnvm.compiler
import vta
import vta.testing

# Load VTA parameters from the vta/config/vta_config.json file
env = vta.get_env()

# Helper to crop an image to a square (224, 224)
# Takes in an Image object, returns an Image object
def thumbnailify(image, pad=15):
    w, h = image.size
    crop = ((w-h)//2+pad, pad, h+(w-h)//2-pad, h-pad)
    image = image.crop(crop)
    image = image.resize((224, 224))
    return image

# Helper function to read in image
# Takes in Image object, returns an ND array
def process_image(image):
    # Convert to neural network input format
    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]

    return tvm.nd.array(image.astype("float32"))

# Classification helper function
# Takes in the graph runtime, and an image, and returns top result and time
def classify(m, image):
    m.set_input('data', image)
    timer = m.module.time_evaluator("run", ctx, number=1)
    tcost = timer()
    tvm_output = m.get_output(0)
    top = np.argmax(tvm_output.asnumpy()[0])
    tcost = "t={0:.2f}s".format(tcost.mean)
    return tcost + " {}".format(synset[top])

# Helper function to compile the NNVM graph
# Takes in a path to a graph file, params file, and device target
# Returns the NNVM graph object, a compiled library object, and the params dict
def generate_graph(graph_fn, params_fn, device="vta"):
    # Measure build start time
    build_start = time.time()

    # Derive the TVM target
    target = tvm.target.create("llvm -device={}".format(device))

    # Derive the LLVM compiler flags
    # When targetting the Pynq, cross-compile to ARMv7 ISA
    if env.TARGET == "sim":
        target_host = "llvm"
    elif env.TARGET == "pynq":
        target_host = "llvm -mtriple=armv7-none-linux-gnueabihf -mcpu=cortex-a9 -mattr=+neon"

    # Load the ResNet-18 graph and parameters
    sym = nnvm.graph.load_json(open(graph_fn).read())
    params = nnvm.compiler.load_param_dict(open(params_fn, 'rb').read())

    # Populate the shape and data type dictionary
    shape_dict = {"data": (1, 3, 224, 224)}
    dtype_dict = {"data": 'float32'}
    shape_dict.update({k: v.shape for k, v in params.items()})
    dtype_dict.update({k: str(v.dtype) for k, v in params.items()})

    # Apply NNVM graph optimization passes
    sym = vta.graph.clean_cast(sym)
    sym = vta.graph.clean_conv_fuse(sym)
    if target.device_name == "vta":
        assert env.BLOCK_IN == env.BLOCK_OUT
        sym = vta.graph.pack(sym, shape_dict, env.BATCH, env.BLOCK_OUT)

    # Compile NNVM graph
    with nnvm.compiler.build_config(opt_level=3):
        if target.device_name != "vta":
            graph, lib, params = nnvm.compiler.build(
                sym, target, shape_dict, dtype_dict,
                params=params, target_host=target_host)
        else:
            with vta.build_config():
                graph, lib, params = nnvm.compiler.build(
                    sym, target, shape_dict, dtype_dict,
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
    print("ResNet-18 inference graph built in {0:.2f}s!".format(build_time))

    return graph, lib, params


######################################################################
# Download ResNet Model
# --------------------------------------------
# Download the necessary files to run ResNet-18.
#

# Obtain ResNet model and download them into _data dir
url = "https://github.com/uwsaml/web-data/raw/master/vta/models/"
categ_fn = 'synset.txt'
graph_fn = 'resnet18_qt8.json'
params_fn = 'resnet18_qt8.params'

# Create data dir
data_dir = "_data/"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Download files
for file in [categ_fn, graph_fn, params_fn]:
    if not os.path.isfile(file):
        download(os.path.join(url, file), os.path.join(data_dir, file))

# Read in ImageNet Categories
synset = eval(open(os.path.join(data_dir, categ_fn)).read())

# Download pre-tuned op parameters of conv2d for ARM CPU used in VTA
autotvm.tophub.check_backend('vta')


######################################################################
# Setup the Pynq Board's RPC Server
# ---------------------------------
# Build the RPC server's VTA runtime and program the Pynq FPGA.

# Measure build start time
reconfig_start = time.time()

# We read the Pynq RPC host IP address and port number from the OS environment
host = os.environ.get("VTA_PYNQ_RPC_HOST", "192.168.2.99")
port = int(os.environ.get("VTA_PYNQ_RPC_PORT", "9091"))

# We configure both the bitstream and the runtime system on the Pynq
# to match the VTA configuration specified by the vta_config.json file.
if env.TARGET == "pynq":
    # Make sure that TVM was compiled with RPC=1
    assert tvm.module.enabled("rpc")
    remote = rpc.connect(host, port)

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


######################################################################
# Build the ResNet Runtime
# ------------------------
# Build the ResNet graph runtime, and configure the parameters.

# Set ``device=vtacpu`` to run inference on the CPU
# or ``device=vta`` to run inference on the FPGA.
device = "vta"

# Device context
ctx = remote.ext_dev(0) if device == "vta" else remote.cpu(0)

# Build the graph runtime
graph, lib, params = generate_graph(os.path.join(data_dir, graph_fn),
                                    os.path.join(data_dir, params_fn),
                                    device)
m = graph_runtime.create(graph, lib, ctx)

# Set the parameters
m.set_input(**params)

######################################################################
# Run ResNet-18 inference on a sample image
# -----------------------------------------
# Perform image classification on test image.
# You can change the test image URL to any image of your choosing.

# Read in test image
image_url = 'https://homes.cs.washington.edu/~moreau/media/vta/cat.jpg'
# Read in test image
response = requests.get(image_url)
image = Image.open(BytesIO(response.content)).resize((224, 224))
# Show Image
plt.imshow(image)
plt.show()
# Set the input
image = process_image(image)
m.set_input('data', image)

# Perform inference
timer = m.module.time_evaluator("run", ctx, number=1)
tcost = timer()

# Get classification results
tvm_output = m.get_output(0)
top_categories = np.argsort(tvm_output.asnumpy()[0])

# Report top-5 classification results
print("ResNet-18 Prediction #1:", synset[top_categories[-1]])
print("                     #2:", synset[top_categories[-2]])
print("                     #3:", synset[top_categories[-3]])
print("                     #4:", synset[top_categories[-4]])
print("                     #5:", synset[top_categories[-5]])
print("Performed inference in {0:.2f}s".format(tcost.mean))


######################################################################
# Run a Youtube Video Image Classifier
# ------------------------------------
# Perform image classification on test stream on 1 frame every 48 frames.
# Comment the `if False:` out to run the demo

# Early exit - remove for Demo
if False:

    import cv2
    import pafy
    from IPython.display import clear_output

    # Helper to crop an image to a square (224, 224)
    # Takes in an Image object, returns an Image object
    def thumbnailify(image, pad=15):
        w, h = image.size
        crop = ((w-h)//2+pad, pad, h+(w-h)//2-pad, h-pad)
        image = image.crop(crop)
        image = image.resize((224, 224))
        return image

    # 16:16 inches
    plt.rcParams['figure.figsize'] = [16, 16]

    # Stream the video in
    url = "https://www.youtube.com/watch?v=PJlmYh27MHg&t=2s"
    video = pafy.new(url)
    best = video.getbest(preftype="mp4")
    cap = cv2.VideoCapture(best.url)

    # Process one frame out of every 48 for variety
    count = 0
    guess = ""
    while(count<2400):

        # Capture frame-by-frame
        ret, frame = cap.read()

        # Process one every 48 frames
        if count % 48 == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            # Crop and resize
            thumb = np.array(thumbnailify(frame))
            image = process_image(thumb)
            guess = classify(m, image)

            # Insert guess in frame
            frame = cv2.rectangle(thumb,(0,0),(200,0),(0,0,0),50)
            cv2.putText(frame, guess, (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (256,256,256), 1, cv2.LINE_AA)

            plt.imshow(thumb)
            plt.axis('off')
            plt.show()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            clear_output(wait=True)

        count += 1

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
