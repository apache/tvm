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

import os
from tvm import relay
from tvm.contrib.download import download_testdata


################################################
# Utils for downloading and extracting zip files
# ----------------------------------------------
def extract(path):
    import tarfile
    if path.endswith("tgz") or path.endswith("gz"):
        dir_path = os.path.dirname(path)
        tar = tarfile.open(path)
        tar.extractall(path=dir_path)
        tar.close()
    else:
        raise RuntimeError('Could not decompress the file: ' + path)


###################################
# Download TFLite pre-trained model
# ---------------------------------

model_url = "https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz"
model_path = download_testdata(model_url, "mobilenet_v2_1.4_224.tgz", module=['tf', 'official'])
model_dir = os.path.dirname(model_path)
extract(model_path)

# now we have mobilenet_v2_1.4_224.tflite on disk
model_file = os.path.join(model_dir, "mobilenet_v2_1.4_224.tflite")

# get TFLite model from buffer
tflite_model_buf = open(model_file, "rb").read()
try:
    import tflite
    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)


##############################
# Load Neural Network in Relay
# ----------------------------

# TFLite input tensor name, shape and type
input_tensor = "input"
input_shape = (1, 224, 224, 3)
input_dtype = "float32"

# parse TFLite model and convert into Relay computation graph
mod, params = relay.frontend.from_tflite(tflite_model,
                                         shape_dict={input_tensor: input_shape},
                                         dtype_dict={input_tensor: input_dtype})

#############
# Compilation
# -----------

target = 'llvm'

# Build with Relay
with relay.build_config(opt_level=3):
    graph, lib, params = relay.build_module.build(
        mod, target, params=params)

###############################################
# Save the graph, lib and parameters into files
# ---------------------------------------------

lib.export_library("./mobilenet.so")
print('lib export succeefully')

with open("./mobilenet.json", "w") as fo:
   fo.write(graph)

with open("./mobilenet.params", "wb") as fo:
   fo.write(relay.save_param_dict(params))
