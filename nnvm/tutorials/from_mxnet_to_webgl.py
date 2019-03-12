"""
Deploy Deep Learning Models to OpenGL and WebGL
===============================================
**Author**: `Zhixun Tan <https://github.com/phisiart>`_

This example shows how to build a neural network with NNVM python frontend and
generate runtime library for WebGL running in a browser with TVM.
To run this notebook, you need to install tvm and nnvm.
Notice that you need to build tvm with OpenGL.
"""

######################################################################
# Overview
# --------
# In this tutorial, we will download a pre-trained resnet18 model from Gluon
# Model Zoo, and run image classification in 3 different ways:
#
# - Run locally:
#   We will compile the model into a TVM library with OpenGL device code and
#   directly run it locally.
#
# - Run in a browser through RPC:
#   We will compile the model into a JavaScript TVM library with WebGL device
#   code, and upload it to an RPC server that is hosting JavaScript TVM runtime
#   to run it.
#
# - Export a JavaScript library and run in a browser:
#   We will compile the model into a JavaScript TVM library with WebGL device
#   code, combine it with JavaScript TVM runtime, and pack everything together.
#   Then we will run it directly in a browser.
#
from __future__ import print_function

import numpy as np
import tvm
import nnvm.compiler
import nnvm.testing

# This tutorial must be run with OpenGL backend enabled in TVM.
# The NNVM CI does not enable OpenGL yet. But the user can run this script.
opengl_enabled = tvm.module.enabled("opengl")

# To run the local demo, set this flag to True.
run_deploy_local = False

# To run the RPC demo, set this flag to True.
run_deploy_rpc = False

# To run the WebGL deploy demo, set this flag to True.
run_deploy_web = False

######################################################################
# Download a Pre-trained Resnet18 Model
# -------------------------------------
# Here we define 2 functions:
#
# - A function that downloads a pre-trained resnet18 model from Gluon Model Zoo.
#   The model that we download is in MXNet format, we then transform it into an
#   NNVM computation graph.
#
# - A function that downloads a file that contains the name of all the image
#   classes in this model.
#
def load_mxnet_resnet():
    """Load a pretrained resnet model from MXNet and transform that into NNVM
       format.

    Returns
    -------
    net : nnvm.Symbol
        The loaded resnet computation graph.

    params : dict[str -> NDArray]
        The pretrained model parameters.

    data_shape: tuple
        The shape of the input tensor (an image).

    out_shape: tuple
        The shape of the output tensor (probability of all classes).
    """

    print("Loading pretrained resnet model from MXNet...")

    # Download a pre-trained mxnet resnet18_v1 model.
    from mxnet.gluon.model_zoo.vision import get_model
    block = get_model('resnet18_v1', pretrained=True)

    # Transform the mxnet model into NNVM.
    # We want a probability so add a softmax operator.
    sym, params = nnvm.frontend.from_mxnet(block)
    sym = nnvm.sym.softmax(sym)

    print("- Model loaded!")
    return sym, params, (1, 3, 224, 224), (1, 1000)

def download_synset():
    """Download a dictionary from class index to name.
    This lets us know what our prediction actually is.

    Returns
    -------
    synset : dict[int -> str]
        The loaded synset.
    """

    print("Downloading synset...")

    from mxnet import gluon

    url = "https://gist.githubusercontent.com/zhreshold/" + \
          "4d0b62f3d01426887599d4f7ede23ee5/raw/" + \
          "596b27d23537e5a1b5751d2b0481ef172f58b539/" + \
          "imagenet1000_clsid_to_human.txt"
    file_name = "synset.txt"

    gluon.utils.download(url, file_name)
    with open(file_name) as f:
        synset = eval(f.read())

    print("- Synset downloaded!")
    return synset

######################################################################
# Download Input Image
# --------------------
# Here we define 2 functions that prepare an image that we want to perform
# classification on.
#
# - A function that downloads a cat image.
#
# - A function that performs preprocessing to an image so that it fits the
#   format required by the resnet18 model.
#
def download_image():
    """Download a cat image and resize it to 224x224 which fits resnet.

    Returns
    -------
    image : PIL.Image.Image
        The loaded and resized image.
    """

    print("Downloading cat image...")

    from matplotlib import pyplot as plt
    from mxnet import gluon
    from PIL import Image

    url = "https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true"
    img_name = "cat.png"

    gluon.utils.download(url, img_name)
    image = Image.open(img_name).resize((224, 224))

    print("- Cat image downloaded!")

    plt.imshow(image)
    plt.show()

    return image

def transform_image(image):
    """Perform necessary preprocessing to input image.

    Parameters
    ----------
    image : numpy.ndarray
        The raw image.

    Returns
    -------
    image : numpy.ndarray
        The preprocessed image.
    """

    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

######################################################################
# Compile the Model
# -----------------
# Here we define a function that invokes the NNVM compiler.
#
def compile_net(net, target_host, target, data_shape, params):
    """Compiles an NNVM computation graph.

    Parameters
    ----------
    net : nnvm.Graph
        The NNVM computation graph.

    target_host : str
        The target to compile the host portion of the library.

    target : str
        The target to compile the device portion of the library.

    data_shape : tuple
        The shape of the input data (image).

    params : dict[str -> NDArray]
        Model parameters.

    Returns
    -------
    graph : Graph
        The final execution graph.

    libmod : tvm.Module
        The module that comes with the execution graph

    params : dict[str -> NDArray]
        The updated parameters of graph if params is passed.
        This can be different from the params passed in.
    """

    print("Compiling the neural network...")

    with nnvm.compiler.build_config(opt_level=0):
        deploy_graph, lib, deploy_params = nnvm.compiler.build(
            net,
            target_host=target_host,
            target=target,
            shape={"data": data_shape},
            params=params)

    print("- Complilation completed!")
    return deploy_graph, lib, deploy_params

######################################################################
# Demo 1: Deploy Locally
# ----------------------
# In this demo, we will compile the model targetting the local machine.
#
# Then we will demonstrate how to save the compiled model as a shared library
# and load it back.
#
# Finally, we will run the model.
#
def deploy_local():
    """Runs the demo that deploys a model locally.
    """

    # Load resnet model.
    net, params, data_shape, out_shape = load_mxnet_resnet()

    # Compile the model.
    # Note that we specify the the host target as "llvm".
    deploy_graph, lib, deploy_params = compile_net(
        net,
        target_host="llvm",
        target="opengl",
        data_shape=data_shape,
        params=params)

    # Save the compiled module.
    # Note we need to save all three files returned from the NNVM compiler.
    print("Saving the compiled module...")
    from tvm.contrib import util
    temp = util.tempdir()

    path_lib = temp.relpath("deploy_lib.so")
    path_graph_json = temp.relpath("deploy_graph.json")
    path_params = temp.relpath("deploy_param.params")

    lib.export_library(path_lib)
    with open(path_graph_json, "w") as fo:
        fo.write(deploy_graph.json())
    with open(path_params, "wb") as fo:
        fo.write(nnvm.compiler.save_param_dict(deploy_params))

    print("- Saved files:", temp.listdir())

    # Load the module back.
    print("Loading the module back...")
    loaded_lib = tvm.module.load(path_lib)
    with open(path_graph_json) as fi:
        loaded_graph_json = fi.read()
    with open(path_params, "rb") as fi:
        loaded_params = bytearray(fi.read())
    print("- Module loaded!")

    # Run the model! We will perform prediction on an image.
    print("Running the graph...")
    from tvm.contrib import graph_runtime

    module = graph_runtime.create(loaded_graph_json, loaded_lib, tvm.opengl(0))
    module.load_params(loaded_params)

    image = transform_image(download_image())
    input_data = tvm.nd.array(image.astype("float32"), ctx=tvm.opengl(0))

    module.set_input("data", input_data)
    module.run()

    # Retrieve the output.
    out = module.get_output(0, tvm.nd.empty(out_shape, ctx=tvm.opengl(0)))
    top1 = np.argmax(out.asnumpy())
    synset = download_synset()
    print('TVM prediction top-1:', top1, synset[top1])

if run_deploy_local and opengl_enabled:
    deploy_local()

######################################################################
# Demo 2: Deploy the Model to WebGL Remotely with RPC
# -------------------------------------------------------
# Following the steps above, we can also compile the model for WebGL.
# TVM provides rpc module to help with remote deploying.
#
# When we deploy a model locally to OpenGL, the model consists of two parts:
# the host LLVM part and the device GLSL part. Now that we want to deploy to
# WebGL, we need to leverage Emscripten to transform LLVM into JavaScript. In
# order to do that, we will need to specify the host target as
# 'llvm -target=asmjs-unknown-emscripten -system-lib`. Then call Emscripten to
# compile the LLVM binary output into a JavaScript file.
#
# First, we need to manually start an RPC server. Please follow the instructions
# in `tvm/web/README.md`. After following the steps, you should have a web page
# opened in a browser, and a Python script running a proxy.
#
def deploy_rpc():
    """Runs the demo that deploys a model remotely through RPC.
    """
    from tvm import rpc
    from tvm.contrib import util, emscripten

    # As usual, load the resnet18 model.
    net, params, data_shape, out_shape = load_mxnet_resnet()

    # Compile the model.
    # Note that this time we are changing the target.
    # This is because we want to translate the host library into JavaScript
    # through Emscripten.
    graph, lib, params = compile_net(
        net,
        target_host="llvm -target=asmjs-unknown-emscripten -system-lib",
        target="opengl",
        data_shape=data_shape,
        params=params)

    # Now we want to deploy our model through RPC.
    # First we ned to prepare the module files locally.
    print("Saving the compiled module...")

    temp = util.tempdir()
    path_obj = temp.relpath("deploy.bc") # host LLVM part
    path_dso = temp.relpath("deploy.js") # host JavaScript part
    path_gl = temp.relpath("deploy.gl") # device GLSL part
    path_json = temp.relpath("deploy.tvm_meta.json")

    lib.save(path_obj)
    emscripten.create_js(path_dso, path_obj, side_module=True)
    lib.imported_modules[0].save(path_gl)

    print("- Saved files:", temp.listdir())

    # Connect to the RPC server.
    print("Connecting to RPC server...")
    proxy_host = 'localhost'
    proxy_port = 9090
    remote = rpc.connect(proxy_host, proxy_port, key="js")
    print("- Connected to RPC server!")

    # Upload module to RPC server.
    print("Uploading module to RPC server...")
    remote.upload(path_dso, "deploy.dso")
    remote.upload(path_gl)
    remote.upload(path_json)
    print("- Upload completed!")

    # Load remote library.
    print("Loading remote library...")
    fdev = remote.load_module("deploy.gl")
    fhost = remote.load_module("deploy.dso")
    fhost.import_module(fdev)
    rlib = fhost
    print("- Remote library loaded!")

    ctx = remote.opengl(0)

    # Upload the parameters.
    print("Uploading parameters...")
    rparams = {k: tvm.nd.array(v, ctx) for k, v in params.items()}
    print("- Parameters uploaded!")

    # Create the remote runtime module.
    print("Running remote module...")
    from tvm.contrib import graph_runtime
    module = graph_runtime.create(graph, rlib, ctx)

    # Set parameter.
    module.set_input(**rparams)

    # Set input data.
    input_data = np.random.uniform(size=data_shape)
    module.set_input('data', tvm.nd.array(input_data.astype('float32')))

    # Run.
    module.run()
    print("- Remote module execution completed!")

    out = module.get_output(0, out=tvm.nd.empty(out_shape, ctx=ctx))
    # Print first 10 elements of output.
    print(out.asnumpy()[0][0:10])

if run_deploy_rpc and opengl_enabled:
    deploy_rpc()

######################################################################
# Demo 3: Deploy the Model to WebGL SystemLib
# -----------------------------------------------
# This time we are not using RPC. Instead, we will compile the model and link it
# with the entire tvm runtime into a single giant JavaScript file. Then we will
# run the model using JavaScript.
#
def deploy_web():
    """Runs the demo that deploys to web.
    """

    import base64
    import json
    import os
    import shutil
    import SimpleHTTPServer, SocketServer

    from tvm.contrib import emscripten

    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(os.getcwd())))
    working_dir = os.getcwd()
    output_dir = os.path.join(working_dir, "resnet")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # As usual, load the resnet18 model.
    net, params, data_shape, out_shape = load_mxnet_resnet()

    # As usual, compile the model.
    graph, lib, params = compile_net(
        net,
        target_host="llvm -target=asmjs-unknown-emscripten -system-lib",
        target="opengl",
        data_shape=data_shape,
        params=params)

    # Now we save the model and link it with the TVM web runtime.
    path_lib = os.path.join(output_dir, "resnet.js")
    path_graph = os.path.join(output_dir, "resnet.json")
    path_params = os.path.join(output_dir, "resnet.params")
    path_data_shape = os.path.join(output_dir, "data_shape.json")
    path_out_shape = os.path.join(output_dir, "out_shape.json")

    lib.export_library(path_lib, emscripten.create_js, options=[
        "-s", "USE_GLFW=3",
        "-s", "USE_WEBGL2=1",
        "-lglfw",
        "-s", "TOTAL_MEMORY=1073741824",
    ])
    with open(path_graph, "w") as fo:
        fo.write(graph.json())
    with open(path_params, "w") as fo:
        fo.write(base64.b64encode(nnvm.compiler.save_param_dict(params)))

    shutil.copyfile(os.path.join(curr_path, "../tvm/web/tvm_runtime.js"),
                    os.path.join(output_dir, "tvm_runtime.js"))
    shutil.copyfile(os.path.join(curr_path, "web/resnet.html"),
                    os.path.join(output_dir, "resnet.html"))

    # Now we want to save some extra files so that we can execute the model from
    # JavaScript.
    # - data shape
    with open(path_data_shape, "w") as fo:
        json.dump(list(data_shape), fo)
    # - out shape
    with open(path_out_shape, "w") as fo:
        json.dump(list(out_shape), fo)
    # - input image
    image = download_image()
    image.save(os.path.join(output_dir, "data.png"))
    # - synset
    synset = download_synset()
    with open(os.path.join(output_dir, "synset.json"), "w") as fo:
        json.dump(synset, fo)

    print("Output files are in", output_dir)

    # Finally, we fire up a simple web server to serve all the exported files.
    print("Now running a simple server to serve the files...")
    os.chdir(output_dir)
    port = 8080
    handler = SimpleHTTPServer.SimpleHTTPRequestHandler
    httpd = SocketServer.TCPServer(("", port), handler)
    print("Please open http://localhost:" + str(port) + "/resnet.html")
    httpd.serve_forever()

if run_deploy_web and opengl_enabled:
    deploy_web()
