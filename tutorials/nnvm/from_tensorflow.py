"""
Compile Tensorflow Models
====================
This article is an introductory tutorial to deploy tensorflow models with NNVM.

For us to begin with, tensorflow module is required to be installed.

A quick solution is to install tensorlfow from

https://www.tensorflow.org/install/install_sources
"""

import nnvm
import tvm
import numpy as np
import os.path
import re

# Tensorflow imports
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util


repo_base = 'https://github.com/srkreddy1238/dmlc_data/raw/master/models/tensorflow/InceptionV1/'
img_name = 'elephant-299.jpg'
image_url = os.path.join(repo_base, img_name)
model_name = 'classify_image_graph_def-with_shapes.pb'
model_url = os.path.join(repo_base, model_name)
map_proto = 'imagenet_2012_challenge_label_map_proto.pbtxt'
map_proto_url = os.path.join(repo_base, map_proto)
lable_map = 'imagenet_synset_to_human_label_map.txt'
lable_map_url = os.path.join(repo_base, lable_map)


######################################################################
# Some helper functions

def _ProcessGraphDefParam(graph_def):
  """Type-checks and possibly canonicalizes `graph_def`."""
  if not isinstance(graph_def, graph_pb2.GraphDef):
    # `graph_def` could be a dynamically-created message, so try a duck-typed
    # approach
    try:
      old_graph_def = graph_def
      graph_def = graph_pb2.GraphDef()
      graph_def.MergeFrom(old_graph_def)
    except TypeError:
      raise TypeError('graph_def must be a GraphDef proto.')
  return graph_def

class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          "./", map_proto)
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          "./", lable_map)
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]

######################################################################
# Download processed tensorflow model
# ---------------------------------------------
# In this section, we download a pretrained Tensorflow model and classify an image.
from mxnet.gluon.utils import download

download(image_url, img_name)
download(model_url, model_name)
download(map_proto_url, map_proto)
download(lable_map_url, lable_map)


######################################################################
# Creates graph from saved graph_def.pb.
with tf.gfile.FastGFile(os.path.join(
        "./", model_name), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    graph = tf.import_graph_def(graph_def, name='')
    graph_def = _ProcessGraphDefParam(graph_def)


######################################################################
# Decode image
from PIL import Image
image = Image.open(img_name).resize((299, 299))

def transform_image(image):
    image = np.array(image)
    return image

x = transform_image(image)
print('x', x.shape)

######################################################################
# Import the graph to NNVM
# -----------------
sym, params = nnvm.frontend.from_tensorflow(graph_def)

######################################################################
# Now compile the graph through NNVM
import nnvm.compiler
target = 'llvm'
shape_dict = {'DecodeJpeg/contents': x.shape}
dtype_dict = {'DecodeJpeg/contents': 'uint8'}
graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, dtype=dtype_dict, params=params)



######################################################################
# Save the compilation output.
"""
lib.export_library("imagenet_tensorflow.so")
with open("imagenet_tensorflow.json", "w") as fo:
    fo.write(graph.json())
with open("imagenet_tensorflow.params", "wb") as fo:
    fo.write(nnvm.compiler.save_param_dict(params))
"""

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
node_lookup = NodeLookup()

top_k = predictions.argsort()[-10:][::-1]
for node_id in top_k:
    human_string = node_lookup.id_to_string(node_id)
    score = predictions[node_id]
    print('%s (score = %.5f)' % (human_string, score))
