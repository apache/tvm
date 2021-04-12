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
import sys
import struct

#
# Disable GPU usage information:
#
#
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

from PIL import Image

import tensorflow as tf

import tvm
import tvm.relay as relay
from tvm.contrib import stm32

TEST_IMAGES = [
        '00.png',
        '01.png',
        '02.png',
        '03.png',
        '04.png',
        '05.png',
        '06.png',
        '07.png',
        '08.png',
        '09.png'
    ]

# Docker: ?

# ==================================================================
#   dump_image
# ==================================================================
def dump_image(filename, image):
    # Flatten image
    image_data = image.flatten()
    outputRaw = [];   #Raw binary format
    for i in range(0, len(image_data)):
        outputRaw.append(struct.pack("<B", int(image_data[i]) & 0xff));

    # Dump image in raw binary format
    f = open(filename, "wb");
    for i in range(0, len(outputRaw)):
        f.write(outputRaw[i]);
    f.close();


# ==================================================================
#   scale_input_data
# ==================================================================
def scale_input_data (input_details, data):
    if input_details['dtype'] == np.uint8 or input_details['dtype'] == np.int8:
        input_scale, input_zero_point = input_details["quantization"]
        print ("== TFLite input quantization: scale={}, zero={}".format(input_scale, input_zero_point))
        data = data / input_scale + input_zero_point
    data = data.astype(input_details["dtype"])
    return data

# ==================================================================
#   scale_output_data
# ==================================================================
def scale_output_data (output_details, data):
    if output_details['dtype'] == np.uint8 or output_details['dtype'] == np.int8:
        output_scale, output_zero_point = output_details["quantization"]
        print ("== TFLite output quantization: scale={}, zero={}".format(output_scale, output_zero_point))
        data = data.astype(np.float32)
        data = (data - output_zero_point) * output_scale
    return data

# ========================================================
#   get_tflite_model
# ========================================================
def get_tflite_model (model_path):

    #
    # Load TFLite model and allocate tensors.
    #
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    #
    # Get input and output tensors.
    #
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #
    # Figure out shapes and
    #
    
    shape_dict = {}
    dtype_dict = {}

    for input in input_details:
        input_name = input['name']
        input_shape = input['shape'].tolist()
        input_dtype = str(np.dtype(input['dtype']))
        shape_dict[input_name] = input_shape
        dtype_dict[input_name] = input_dtype

    #
    # Save the model
    #
        
    #
    # Load the TFLite Model for TVM:
    #
    # https://docs.tvm.ai/tutorials/frontend/from_tflite.html
    # https://jackwish.net/tflite/docs/

    model_buf = open (model_path, "rb").read()

    #
    # Get TFLite model from buffer
    #
    try:
        import tflite
        model = tflite.Model.GetRootAsModel(model_buf, 0)
        assert isinstance(model, tflite.Model)
    except AttributeError:
        import tflite.Model
        model = tflite.Model.Model.GetRootAsModel(model_buf, 0)
        assert isinstance(model, tflite.Model.Model)

    print("TVM: Importing a TFLite model ...")
        
    return model, shape_dict, dtype_dict

# ========================================================
#   extract_tflite_quantization
# ========================================================

def _make_qnn_params (quantization):
    qnn_params = {}
    qnn_params['min'] = quantization.MinAsNumpy()
    qnn_params['max'] = quantization.MaxAsNumpy()
    qnn_params['scale'] = quantization.ScaleAsNumpy()
    qnn_params['zero_point'] = quantization.ZeroPointAsNumpy()
    qnn_params['dim'] = quantization.QuantizedDimension()
    #print("  Quantization: ({}, {}), s={}, z={}, dim={}".format(min, max, scale, zero_point, dim))
    return qnn_params

def extract_tflite_quantization (model):

    assert model.SubgraphsLength() == 1, "only support one subgraph (main subgraph)"

    subgraph = model.Subgraphs(0)

    quantization_info = {}

    # model inputs / outputs
    model_inputs = subgraph.InputsAsNumpy()
    model_outputs = subgraph.OutputsAsNumpy()

    for node_id in model_inputs:
        tensor = subgraph.Tensors(node_id)
        tensor_name = tensor.Name().decode('utf-8')
        tensor_type = tensor.Type()
        #print("== Input[{}]: {} shape={} type={}".format(node_id, tensor_name, tensor.ShapeAsNumpy(), tensor_type))
        dl_tensor_name = stm32.get_input_tensor_name(tensor_name)

        quantization = tensor.Quantization()
        if quantization is not None:
            qnn_params = _make_qnn_params (quantization)
            quantization_info[dl_tensor_name] = qnn_params

    for node_id in model_outputs:
        tensor = subgraph.Tensors(node_id)
        tensor_name = tensor.Name().decode('utf-8')
        tensor_type = tensor.Type()
        #print("== Output[{}]: {} shape={} type={}".format(node_id, tensor_name, tensor.ShapeAsNumpy(), tensor_type))
        #
        # TODO: TVM does not preserve the output tensor names.
        #       Eventually, we should be able to form a valid name.
        #
        dl_tensor_name = stm32.get_output_tensor_name(tensor_name,0)

        quantization = tensor.Quantization()
        if quantization is not None:
            qnn_params = _make_qnn_params (quantization)
            quantization_info[dl_tensor_name] = qnn_params

    return quantization_info

# ========================================================
#   run_tflite_model
# ========================================================
def run_tflite_model (model_path):

    #
    # Load TFLite model and allocate tensors.
    #
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    #
    # Get input and output tensors.
    #
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    #
    # Run test images
    #
    #tf_results = []
    tf_results = np.empty(shape=[len(TEST_IMAGES),10], dtype=np.float)
    for i, filename in enumerate(TEST_IMAGES):
        #filename = '/prj/p_mcd_ais/common/Appli/data/mnist/test_images/01647.png'
        image = Image.open (filename).convert('L')
        image_data = np.array(image, dtype="uint8")
        #
        # Run the TFLite model: channels last
        #
        image_data = image_data.reshape ([1, 28, 28, 1])
        #
        # Dump as file
        #
        #dump_image('img0.bin', image_data)

        #
        # Normalize the input data
        #
        image_data = image_data/255.0

        image_data = scale_input_data (input_details, image_data)
        interpreter.set_tensor(input_details['index'], image_data)
        interpreter.invoke()
        tf_results[i] = interpreter.get_tensor(output_details['index'])
        tf_results[i] = scale_output_data (output_details, tf_results[i])

        print (f'== [{i}] TFLite Output:')
        print (tf_results[i])

    return tf_results

# ========================================================
#   run_tvm_model
# ========================================================
def run_tvm_model (model_name, target_dir):

    tvm_results_name = 'tvm_results.txt'

    #
    # Build the model
    #
    command = f'make MODEL_PATH={target_dir}'
    print (f'{command} ...')
    os.system(command)
    #
    # Run
    #
    for filename in TEST_IMAGES:
        image = Image.open (filename).convert('L')
        image_data = np.array(image, dtype="uint8")
        #
        # Run the TFLite model: channels last
        #
        image_data = image_data.reshape ([1, 28, 28, 1])
        #
        # Dump as file
        #
        outfile = os.path.splitext(filename)[0]+'.bin'
        dump_image(outfile, image_data)
    
    command = f'{target_dir}/{model_name}.exe'
    print (f'{command} ...')
    os.system(command)
    
    tvm_results = np.loadtxt(tvm_results_name)
    #print (f'== tvm_results shape: {tvm_results.shape}')
    
    #
    # Clean temporary image files
    #
    if os.path.exists (tvm_results_name):
        os.remove(tvm_results_name)
        
    for filename in TEST_IMAGES:
        outfile = os.path.splitext(filename)[0]+'.bin'
        if os.path.exists (outfile):
            os.remove(outfile)

    return tvm_results
    
# ========================================================
#   test_mnist_fp
# ========================================================
def test_mnist_fp():

    model_name = 'network'
    target_dir = 'mnist_fp_gen'
    
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    model_path = os.path.join(curr_path, 'mnist.tflite')
    
    model, shape_dict, dtype_dict = get_tflite_model (model_path)
    mod, params = relay.frontend.from_tflite(model, shape_dict, dtype_dict)

    #
    # Build a TVM C module for the ARM CPU (without compiling the kernels 
    # library to the object code form):
    #
    target = "c -device=arm_cpu"
    opt_level=3

    with tvm.transform.PassContext(opt_level=opt_level, config={'tir.disable_vectorize': True}):
        rt_module = relay.build(mod,
                                target=target,
                                params=params
        )

    #
    # Export model library format
    #
    mlf_tar_path = "mnist_fp_lib.tar"
    import tvm.micro as micro
    micro.export_model_library_format(rt_module, mlf_tar_path)

    emitter = stm32.CodeEmitter()
    quantization = extract_tflite_quantization (model)
    emitter.parse_library_format (mlf_tar_path, quantization)
    emitter.emit_code (target_dir, model_name)

    #
    # Results
    #
    tf_results = run_tflite_model (model_path)
    tvm_results = run_tvm_model (model_name, target_dir)

    #tvm.testing.assert_allclose(tf_results.asnumpy(), tvm_results.asnumpy())
    np.allclose(tf_results, tvm_results)

if __name__ == "__main__":
    test_mnist_fp()
