#!/usr/bin/env python3

import os
import sys
import argparse

#
# Disable GPU usage information:
#
#
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from collections import OrderedDict
import numpy as np
import json

import tensorflow as tf

import onnxruntime as rt
import tvm
from tvm import relay
from tvm.contrib.target.onnx import to_onnx

class STM32AIException(Exception):
    """stm32ai Exception"""

# =========================================================
#   partition_graph
# =========================================================
def partition_graph(mod):
    """Alter the layout of the input graph.

    Parameters
    ----------
    mod : tvm.relay.Module
        The relay module to convert.

    Returns
    -------
    mod : tvm.relay.Module
        The converted module.
    """

    # Assume for the time being that graphs only have
    # conv2d as heavily-sensitive operators.
    desired_layouts = {
        "nn.conv2d": ["NCHW", "OIHW"],
        "qnn.conv2d": ["NCHW", "OIHW"],
    }

    # Convert the layout of the graph where possible.
    seq = tvm.transform.Sequential(
        [
            relay.transform.RemoveUnusedFunctions(),
            relay.transform.ConvertLayout(desired_layouts),
            relay.transform.AnnotateTarget("onnx"),
            relay.transform.PartitionGraph()
        ]
    )

    with tvm.transform.PassContext(opt_level=3):
        try:
            return seq(mod)
        except Exception as err:
            raise STM32AIException(
                "Error converting layout to {0}: {1}".format(desired_layout, str(err))
            )

# =========================================================
#   get_data
# =========================================================
def get_data(in_data_shapes, in_data_dtypes):
    in_data = OrderedDict()
    for shape_name, shape in in_data_shapes.items():
        for dtype_name, dtype in in_data_dtypes.items():
            if dtype_name == shape_name:
                in_data[shape_name] = np.random.uniform(size=shape).astype(dtype)
                break
        if shape_name not in in_data.keys():
            raise STM32AIException("Shape and dtype dictionaries do not fit.")
            
    return in_data

# =========================================================
#   run_relay
# =========================================================
def run_relay(mod, params, in_data):
    target = "llvm"
    dev = tvm.device("llvm", 0)
    intrp = relay.create_executor("graph", mod, device=dev, target=target)
    in_data = [tvm.nd.array(value) for value in in_data.values()]
    return intrp.evaluate()(*in_data, **params).asnumpy()

# =========================================================
#   func_to_onnx
# =========================================================
def func_to_onnx(mod, params, name):

    print (f'== Relay Module: \n{mod}')
    
    onnx_model = to_onnx(mod, params, name, path=None)
    return onnx_model.SerializeToString()

# =========================================================
#   run_onnx
# =========================================================
def run_onnx(mod, params, name, input_data):
    onnx_model = func_to_onnx(mod, params, name)
    sess = rt.InferenceSession(onnx_model)
    input_names = {}
    for input, data in zip(sess.get_inputs(), input_data):
        input_names[input.name] = data
    output_names = [output.name for output in sess.get_outputs()]
    res = sess.run(output_names, input_names)
    return res[0]

# =========================================================
#   _verify_results
# =========================================================
def _verify_results(mod, params, in_data):
    print (f'== Executing relay interpreter ...')
    a = run_relay(mod, params, in_data)
    print (f'== a = {a}')
    print (f'== Executing ONNX model ...')
    b = run_onnx(mod, params, "test_resent", in_data.values())
    print (f'== b = {b}')
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)

# =========================================================
#   get_tflite_model
# =========================================================
def get_tflite_model(model_path):

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
        input_name = input["name"]
        input_shape = input["shape"].tolist()
        input_dtype = str(np.dtype(input["dtype"]))
        shape_dict[input_name] = input_shape
        dtype_dict[input_name] = input_dtype

    #
    # Load the TFLite Model for TVM:
    #
    # https://docs.tvm.ai/tutorials/frontend/from_tflite.html
    # https://jackwish.net/tflite/docs/

    model_buf = open(model_path, "rb").read()

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
    
# =========================================================
#   main
# =========================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-model", type=str, required=True, help="The TFLite model to compile")
    parser.add_argument(
        "-target-dir", type=str, help="The directory for storing the generated implementation"
    )

    args = parser.parse_args()

    model_path = args.model
    target_dir = args.target_dir

    #
    # Extract the model name
    #
    model_file = os.path.basename(model_path)
    print("=== TVM: Model name: {}".format(model_file))
    model_file_ext = os.path.splitext(model_file)
    assert model_file_ext[1] == ".tflite"

    model_name = model_file_ext[0]

    if not target_dir:
        target_dir = model_file_ext[0] + "_aton"

    model, shape_dict, dtype_dict = get_tflite_model(model_path)
        
    #
    # Import the model with relay
    #
    print("=== TVM: Importing the model.")
    mod, params = relay.frontend.from_tflite(model, shape_dict, dtype_dict)
    print("=== TVM: Compiling the TFLite model ...")

    print(f'== {shape_dict}')
    print(f'== {dtype_dict}')

    mod = partition_graph(mod)

    print (f'== partitioned module: \n{mod}')
    
    #in_data_shapes = OrderedDict({"data": (1, 3, 224, 224)})
    #in_data_shapes = OrderedDict(shape_dict)
    #print(f'== {in_data_shapes}')

    #dtype_list = list(dtype_dict)
    #dtype_list = dtype_dict.values()
    
    in_data = get_data(shape_dict, dtype_dict)
    _verify_results(mod, params, in_data)

    target = "llvm"
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["FuseOps"]):
        tvm_module = relay.build(mod, target)

    print (f'== TVM Module: {tvm_module}')
    graph = tvm_module.get_json()
    if not isinstance(graph, (str,)):
        try:
            graph = graph._tvm_graph_json()
        except AttributeError:
            raise STM32AIException("Type %s is not supported" % type(graph))

    graph_ = json.loads(graph)
    params_ = tvm_module.get_params()
    lib_ = tvm_module.get_lib()

    print (f'  Lib type: {lib_.type_key}')
    assert lib_.type_key == "metadata"
    
    for m in lib_.imported_modules:
        print (f'  sub-module: {m}')
    
    #assert tvm_module.type_key == "metadata"
    #assert tvm_module.imported_modules[0].type_key == "llvm"
    #assert tvm_module.imported_modules[0].get_source()
    #assert tvm_module.imported_modules[1].type_key == "onnx"
    #assert tvm_module.imported_modules[1].get_source()
