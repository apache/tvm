"""
Compile Tensorflow Models
=========================
This article is an introductory tutorial to deploy tensorflow models with NNVM.

For us to begin with, tensorflow module is required to be installed.

A quick solution is to install tensorlfow from

https://www.tensorflow.org/install/install_sources
"""

import nnvm
import tvm
import numpy as np
import os.path

# Tensorflow imports
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util

import nnvm.testing.tf

repo_base = 'https://github.com/dmlc/web-data/raw/master/tensorflow/models/InceptionV1/'
img_name = 'elephant-299.jpg'
image_url = os.path.join(repo_base, img_name)
model_name = 'classify_image_graph_def-with_shapes.pb'
model_url = os.path.join(repo_base, model_name)
map_proto = 'imagenet_2012_challenge_label_map_proto.pbtxt'
map_proto_url = os.path.join(repo_base, map_proto)
lable_map = 'imagenet_synset_to_human_label_map.txt'
lable_map_url = os.path.join(repo_base, lable_map)


######################################################################
# Download processed tensorflow model
# -----------------------------------
# In this section, we download a pretrained Tensorflow model and classify an image.
from mxnet.gluon.utils import download

download(image_url, img_name)
download(model_url, model_name)
download(map_proto_url, map_proto)
download(lable_map_url, lable_map)


######################################################################
# Creates graph from saved graph_def.pb.
# --------------------------------------

with tf.gfile.FastGFile(os.path.join(
        "./", model_name), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    graph = tf.import_graph_def(graph_def, name='')
    # Call the utility to import the graph definition into default graph.
    graph_def = nnvm.testing.tf.ProcessGraphDefParam(graph_def)


######################################################################
# Decode image
# ------------
from PIL import Image
image = Image.open(img_name).resize((299, 299))

def transform_image(image):
    image = np.array(image)
    return image

x = transform_image(image)

######################################################################
# Import the graph to NNVM
# ------------------------
sym, params = nnvm.frontend.from_tensorflow(graph_def)

######################################################################
# Now compile the graph through NNVM
import nnvm.compiler
target = 'llvm'
shape_dict = {'DecodeJpeg/contents': x.shape}
dtype_dict = {'DecodeJpeg/contents': 'uint8'}
graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, dtype=dtype_dict, params=params)

######################################################################
# Execute the portable graph on TVM
# ---------------------------------
# Now, we would like to reproduce the same forward computation using TVM.
from tvm.contrib import graph_runtime
ctx = tvm.cpu(0)
dtype = 'uint8'
m = graph_runtime.create(graph, lib, ctx)
# set inputs
m.set_input('DecodeJpeg/contents', tvm.nd.array(x.astype(dtype)))
m.set_input(**params)
# execute
m.run()
# get outputs
tvm_output = m.get_output(0, tvm.nd.empty(((1, 1008)), 'float32'))


######################################################################
# Process the output to human readable
# ------------------------------------
predictions = tvm_output.asnumpy()
predictions = np.squeeze(predictions)

# Creates node ID --> English string lookup.
node_lookup = nnvm.testing.tf.NodeLookup(label_lookup_path=os.path.join("./", map_proto),
                                         uid_lookup_path=os.path.join("./", lable_map))

top_k = predictions.argsort()[-5:][::-1]
for node_id in top_k:
    human_string = node_lookup.id_to_string(node_id)
    score = predictions[node_id]
    print('%s (score = %.5f)' % (human_string, score))

######################################################################
# Run the same graph with tensorflow and dump output.
# ---------------------------------------------------

def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(model_name, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        graph = tf.import_graph_def(graph_def, name='')
        # Call the utility to import the graph definition into default graph.
        graph_def = nnvm.testing.tf.ProcessGraphDefParam(graph_def)

def run_inference_on_image(image):
    """Runs inference on an image.

    Parameters
    ----------
    image: String
        Image file name.

    Returns
    -------
        Nothing
    """
    if not tf.gfile.Exists(image):
        tf.logging.fatal('File does not exist %s', image)
    image_data = tf.gfile.FastGFile(image, 'rb').read()

    # Creates graph from saved GraphDef.
    create_graph()

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})

        predictions = np.squeeze(predictions)

        # Creates node ID --> English string lookup.
        node_lookup = nnvm.testing.tf.NodeLookup(label_lookup_path=os.path.join("./", map_proto),
                                                 uid_lookup_path=os.path.join("./", lable_map))

        top_k = predictions.argsort()[-5:][::-1]
        print ("===== TENSORFLOW RESULTS =======")
        for node_id in top_k:
            human_string = node_lookup.id_to_string(node_id)
            score = predictions[node_id]
            print('%s (score = %.5f)' % (human_string, score))

run_inference_on_image (img_name)
