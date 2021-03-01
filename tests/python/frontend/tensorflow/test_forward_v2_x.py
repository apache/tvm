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
# pylint: disable=import-self, invalid-name, unused-argument
"""
Tensorflow testcases
====================
This article is a test script to test tensorflow operator with Relay.
"""
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import graph_util
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import variables
from tensorflow.python.training.tracking import tracking
from tensorflow.python.tools import freeze_graph
from tensorflow.python.saved_model.save import save
from tensorflow.python.saved_model.load import load
from tensorflow.python.framework import convert_to_constants
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import device_properties_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.grappler import cluster as gcluster
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.training.saver import export_meta_graph

from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import init_ops
from distutils.version import LooseVersion
import tvm
from tvm import relay
import tvm.relay.testing.tf as tf_testing


#######################################################################
# Generic run functions for TVM & tensorflow
# ------------------------------------------


def convert_to_list(x):
    if not isinstance(x, list):
        x = [x]
    return x


tf_dtypes = {
    "float32": tf.float32,
    "float16": tf.float16,
    "float64": tf.float64,
    "int32": tf.int32,
    "uint8": tf.uint8,
    "int8": tf.int8,
    "int16": tf.int16,
    "uint16": tf.uint16,
    "int64": tf.int64,
}


def vmobj_to_list(o):
    if isinstance(o, tvm.nd.NDArray):
        return [o.asnumpy().tolist()]
    elif isinstance(o, tvm.container.ADT):
        result = []
        for f in o:
            result.extend(vmobj_to_list(f))
        return result
    elif isinstance(o, tvm.relay.backend.interpreter.ConstructorValue):
        if o.constructor.name_hint == "Cons":
            tl = vmobj_to_list(o.fields[1])
            hd = vmobj_to_list(o.fields[0])
            hd.extend(tl)
            return hd
        elif o.constructor.name_hint == "Nil":
            return []
        elif "tensor_nil" in o.constructor.name_hint:
            return [0]
        elif "tensor" in o.constructor.name_hint:
            return [o.fields[0].asnumpy()]
        else:
            raise RuntimeError("Unknown object type: %s" % o.constructor.name_hint)
    else:
        raise RuntimeError("Unknown object type: %s" % type(o))


def run_tvm_graph(
    graph_def,
    input_data,
    input_node,
    num_output=1,
    target="llvm",
    out_names=None,
    opt_level=3,
    mode="graph_runtime",
    cuda_layout="NCHW",
):
    """ Generic function to compile on relay and execute on tvm """
    input_data = convert_to_list(input_data)
    input_node = convert_to_list(input_node)
    layout = None
    if target == "cuda":
        layout = cuda_layout
    target_host = None
    shape_dict = {e: i.shape for e, i in zip(input_node, input_data)}
    mod, params = relay.frontend.from_tensorflow(
        graph_def, layout=layout, shape=shape_dict, outputs=out_names
    )
    if mode in ["debug", "vm"]:
        ex = relay.create_executor(mode, mod=mod, ctx=tvm.cpu(), target="llvm")
        inputs = []
        for param in mod["main"].params:
            found = False
            for i, n in enumerate(input_node):
                if n == param.name_hint:
                    found = True
                    inputs.append(tvm.nd.array(input_data[i]))
                    break
            # Interpreter doesn't bind constants, so still need to find in params
            if not found:
                inputs.append(tvm.nd.array(params[param.name_hint]))
        result = ex.evaluate()(*inputs)
        return vmobj_to_list(result)
    else:
        with relay.build_config(opt_level=opt_level):
            graph, lib, params = relay.build(mod, target, target_host, params)

        ctx = tvm.context(target, 0)
        from tvm.contrib import graph_runtime

        m = graph_runtime.create(graph, lib, ctx)
        # set inputs
        for e, i in zip(input_node, input_data):
            m.set_input(e, tvm.nd.array(i))

        m.set_input(**params)
        # execute
        m.run()
        # get outputs
        assert out_names is None or num_output == len(
            out_names
        ), "out_names: {} num_output: {}".format(out_names, num_output)
        tvm_output_list = [m.get_output(i).asnumpy() for i in range(num_output)]
        return tvm_output_list


# Ref. taken from Tensorflow JS
def _build_signature_def(frozen_graph, input_nodes, output_nodes):
    signature = meta_graph_pb2.SignatureDef()
    for input_tensor in input_nodes:
        op_name = input_tensor.name.split(":")[0]
        # The graph freezing may turn the original inputs into constants, or remove
        # them from the graph, so we need to ignore those.
        try:
            op = frozen_graph.get_operation_by_name(op_name)
            if op.type != "Const":
                signature.inputs[input_tensor.name].name = input_tensor.name
                signature.inputs[input_tensor.name].dtype = input_tensor.dtype.as_datatype_enum
                signature.inputs[input_tensor.name].tensor_shape.CopyFrom(
                    input_tensor.shape.as_proto()
                )
        except KeyError:
            # The original input was removed when the graph was frozen.
            continue
    for output_tensor in output_nodes:
        if hasattr(output_tensor, "name"):
            signature.outputs[output_tensor.name].name = output_tensor.name
            signature.outputs[output_tensor.name].dtype = output_tensor.dtype.as_datatype_enum
            signature.outputs[output_tensor.name].tensor_shape.CopyFrom(
                output_tensor.shape.as_proto()
            )
        else:  # just the tensor name string array
            signature.outputs[output_tensor].name = output_tensor
    return signature


def get_cluster():
    """Grappler optimization configuration for GPU."""
    named_device = device_properties_pb2.NamedDevice()
    named_device.name = "/GPU:0"
    named_device.properties.type = "GPU"
    named_device.properties.environment["architecture"] = "4"
    cluster = gcluster.Cluster(devices=[named_device])
    return cluster


def _run_grappler(config, graph_def, graph, signature_def):
    meta_graph = export_meta_graph(graph_def=graph_def, graph=graph)

    meta_graph.signature_def["not_used_key"].CopyFrom(signature_def)

    return tf_optimizer.OptimizeGraph(config, meta_graph, cluster=get_cluster())


def compare_tf_with_tvm_v2(
    in_data,
    concrete_func,
    no_gpu=False,
    opt_level=3,
    mode="graph_runtime",
    cuda_layout="NCHW",
    tf_out=None,
    atol=1e-5,
    rtol=1e-5,
):
    """Generic function to execute and compate tensorflow and tvm output on given concrete function

    Parameters:
    -----------

    in_data : A list of tensorflow tensor objects.
        Reference inputs for the graph.

    concrete_func : Tensorflow concrete function of the graph.
        Tensorflow function definition.

    no_gpu : bool
        Flag to indicate not to run the testcase on GPU.

    opt_level : Int
        Optimization level for TVM build process.

    mode : Str
        Execution mode of TVM.

    cuda_layout : Str
        The layout to be used on CUDA targets.

    tf_out : Tensorflow Tensor
        The reference Tensorflow output to compare with.

    atol, rtol : float
        Output comparison constraints.

    """
    if tf_out == None:
        tf_out = concrete_func(*in_data)
        if not type(tf_out) == list and not type(tf_out) == tuple:
            tf_out = [tf_out]

    graph = convert_to_constants.convert_variables_to_constants_v2(concrete_func).graph
    signature = _build_signature_def(graph, concrete_func.inputs, concrete_func.outputs)
    graph_def = graph.as_graph_def()

    # Some optimization if needed
    config = config_pb2.ConfigProto()
    rewriter_config = config.graph_options.rewrite_options
    rewriter_config.optimizers[:] = [
        "debug_stripper",
        "pruning",
        "constfold",
        "arithmetic",
        "dependency",
        "pruning",
        "constfold",
        "arithmetic",
        "dependency",
    ]
    rewriter_config.optimizers[:] = [
        "debug_stripper",
        "arithmetic",
        "dependency",
        "arithmetic",
        "dependency",
    ]

    optimized_graph = _run_grappler(config, graph_def, graph, signature)

    output_node_names = []
    for output_tensor in concrete_func.outputs:
        output_node_names.append(output_tensor.name.split(":")[0])

    input_node_names = []
    for input_tensor in concrete_func.inputs:
        input_node_names.append(input_tensor.name.split(":")[0])

    print("Input Names:", input_node_names)
    print("Output Names:", output_node_names)

    for device in ["llvm", "cuda"]:
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            continue
        if no_gpu and device == "cuda":
            continue

        # print("Graph:", optimized_graph)
        for node in optimized_graph.node:
            print("Node:", node.name, "   Op:", node.op)

        tvm_output = run_tvm_graph(
            optimized_graph,
            in_data,
            input_node_names,
            target=device,
            out_names=output_node_names,
            num_output=len(output_node_names),
            opt_level=opt_level,
            mode=mode,
            cuda_layout=cuda_layout,
        )
        # print("TVM Out :", tvm_output)
        # print("TF Out:", tf_out)

        for i in range(len(output_node_names)):
            tvm.testing.assert_allclose(tf_out[i], tvm_output[i], atol=atol, rtol=rtol)


def is_gpu_available():
    from tensorflow.python.client import device_lib

    local_device_protos = device_lib.list_local_devices()
    gpu_list = [x.name for x in local_device_protos if x.device_type == "GPU"]
    if len(gpu_list) > 0:
        print("Tensorflow GPU:", gpu_list)
        return True
    else:
        return False


#######################################################################
# Helper Function
# ---------------


def _test_forward_op(api, param):
    root = tracking.AutoTrackable()
    if len(param["tensor_args"]) == 1:
        root.f = def_function.function(lambda x: api(x, **(param["ext_args"])))
    if len(param["tensor_args"]) == 2:
        root.f = def_function.function(lambda x, y: api(x, y, **(param["ext_args"])))
    func_params = []
    input_data = []
    for arg in param["tensor_args"]:
        func_params.append(tensor_spec.TensorSpec(arg["shape"], arg["dtype"]))
        input_data.append(
            tf.random.uniform(arg["shape"], arg["min_val"], arg["max_val"], dtype=arg["dtype"])
        )

    concrete_func = root.f.get_concrete_function(*func_params)
    compare_tf_with_tvm_v2(input_data, concrete_func)


#######################################################################
# Pooling
# -------


def _test_pooling_iteration(input_shape, **kwargs):
    """ One iteration of pool operation with given shapes and attributes """
    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x: tf.nn.pool(x, **kwargs))
    concrete_func = root.f.get_concrete_function(
        tensor_spec.TensorSpec(input_shape, dtypes.float32)
    )

    input_data = tf.random.uniform(input_shape, minval=-1, maxval=1, dtype=dtypes.float32)
    compare_tf_with_tvm_v2([input_data], concrete_func)


def _test_pooling(input_shape, **kwargs):
    _test_pooling_iteration(input_shape, **kwargs)

    if is_gpu_available():
        if len(input_shape) == 4:
            input_shape = [input_shape[ii] for ii in (0, 3, 1, 2)]
            kwargs["data_format"] = "NCHW"
            _test_pooling_iteration(input_shape, **kwargs)


def test_forward_pooling():
    """ Pooling """
    # TensorFlow only supports NDHWC for max_pool3d on CPU
    for pool_type in ["AVG", "MAX"]:
        # NDHWC is the default layout for max_pool3d and avg_pool3d in TensorFlow
        _test_pooling(
            input_shape=[1, 3, 32, 32, 32],
            window_shape=[2, 2, 2],
            padding="VALID",
            pooling_type=pool_type,
            dilations=[1, 1, 1],
            strides=[2, 2, 2],
        )

        _test_pooling(
            input_shape=[1, 3, 32, 32, 32],
            window_shape=[1, 1, 1],
            padding="SAME",
            pooling_type=pool_type,
            dilations=[1, 1, 1],
            strides=[1, 1, 1],
        )

        _test_pooling(
            input_shape=[1, 3, 32, 32, 32],
            window_shape=[2, 2, 2],
            padding="SAME",
            pooling_type=pool_type,
            dilations=[1, 1, 1],
            strides=[2, 2, 2],
        )

        # test cases for max_pool3d & avg_pool3d with layout NCDHW
        # TensorFlow pool3d  doesn't support NCDHW on cpu
        if is_gpu_available():
            _test_pooling(
                input_shape=[1, 3, 32, 32, 32],
                window_shape=[1, 1, 1],
                padding="SAME",
                pooling_type=pool_type,
                dilations=[1, 1, 1],
                strides=[1, 1, 1],
                data_format="NCDHW",
            )

            _test_pooling(
                input_shape=[1, 3, 32, 32, 32],
                window_shape=[2, 2, 2],
                padding="VALID",
                pooling_type=pool_type,
                dilations=[1, 1, 1],
                strides=[2, 2, 2],
                data_format="NCDHW",
            )

        _test_pooling(
            input_shape=[2, 9, 10, 2],
            window_shape=[1, 1],
            padding="SAME",
            pooling_type=pool_type,
            dilations=[1, 1],
            strides=[1, 1],
        )

        _test_pooling(
            input_shape=[2, 10, 9, 2],
            window_shape=[1, 1],
            padding="SAME",
            pooling_type=pool_type,
            dilations=[1, 1],
            strides=[1, 1],
        )

        _test_pooling(
            input_shape=[2, 9, 10, 2],
            window_shape=[2, 1],
            padding="SAME",
            pooling_type=pool_type,
            dilations=[1, 1],
            strides=[1, 1],
        )

        _test_pooling(
            input_shape=[2, 10, 9, 2],
            window_shape=[2, 3],
            padding="SAME",
            pooling_type=pool_type,
            dilations=[1, 1],
            strides=[2, 1],
        )

        """ TODO
        # Tests involving SpaceToBatchND
        _test_pooling(input_shape=[1, 1, 2, 1],
                      window_shape=[1, 1],
                      padding='VALID',
                      pooling_type=pool_type,
                      dilations=[1, 2])

        _test_pooling(input_shape=[1, 2, 1],
                      window_shape=[1],
                      padding='VALID',
                      pooling_type=pool_type,
                      dilations=[2])
        """


#######################################################################
# Convolution
# -----------


def _test_convolution(
    opname,
    tensor_in_sizes,
    filter_in_sizes,
    dilations,
    strides,
    padding,
    data_format,
    deconv_output_shape=[],
):
    """ One iteration of convolution with given shapes and attributes """

    filter_data = tf.convert_to_tensor(
        np.random.uniform(-1, 1, size=filter_in_sizes).astype("float32")
    )
    if data_format == "NHWC":
        strides = [1] + strides + [1]
        dilations = [1] + dilations + [1]
    else:
        strides = [1, 1] + strides
        dilations = [1, 1] + dilations

    root = tracking.AutoTrackable()

    if opname == "conv":
        root.f = def_function.function(
            lambda x: tf.nn.conv2d(
                x,
                filter_data,
                strides=strides,
                dilations=dilations,
                padding=padding,
                data_format=data_format,
            )
        )
    elif opname == "conv_transpose":
        root.f = def_function.function(
            lambda x: tf.nn.conv2d_transpose(
                x,
                filter_data,
                output_shape=deconv_output_shape,
                strides=strides,
                padding=padding,
                data_format=data_format,
            )
        )
    else:
        # TODO : Support for tf.nn.depthwise_conv2d in TF 2.0
        root.f = def_function.function(
            lambda x: tf.compat.v1.nn.depthwise_conv2d_native(
                x,
                filter_data,
                strides=strides,
                dilations=dilations,
                padding=padding,
                data_format=data_format,
            )
        )

    concrete_func = root.f.get_concrete_function(
        tensor_spec.TensorSpec(tensor_in_sizes, dtypes.float32)
    )
    input_data = tf.random.uniform(tensor_in_sizes, minval=-1, maxval=1, dtype=dtypes.float32)
    compare_tf_with_tvm_v2([input_data], concrete_func)


def test_forward_convolution():
    if is_gpu_available():
        _test_convolution("conv", [4, 176, 8, 8], [1, 1, 176, 32], [1, 1], [1, 1], "SAME", "NCHW")
        _test_convolution("conv", [4, 19, 17, 17], [3, 3, 19, 19], [1, 1], [2, 2], "VALID", "NCHW")
        _test_convolution("conv", [4, 124, 17, 17], [1, 1, 124, 19], [1, 1], [1, 1], "SAME", "NCHW")
        _test_convolution("conv", [4, 12, 17, 17], [3, 3, 12, 32], [1, 1], [2, 2], "VALID", "NCHW")
        _test_convolution(
            "depthwise", [4, 176, 8, 8], [1, 1, 176, 1], [1, 1], [1, 1], "SAME", "NCHW"
        )
        _test_convolution(
            "depthwise", [4, 19, 17, 17], [3, 3, 19, 1], [1, 1], [2, 2], "VALID", "NCHW"
        )
        _test_convolution(
            "depthwise", [4, 124, 17, 17], [1, 1, 124, 1], [1, 1], [1, 1], "SAME", "NCHW"
        )
        _test_convolution(
            "depthwise", [4, 12, 17, 17], [3, 3, 12, 1], [1, 1], [2, 2], "VALID", "NCHW"
        )
        _test_convolution(
            "depthwise", [4, 12, 17, 17], [3, 3, 12, 2], [1, 1], [2, 2], "VALID", "NCHW"
        )
        """ TODO
        _test_convolution('conv_transpose', [4, 32, 8, 8], [1, 1, 176, 32], [1, 1], [1, 1], 'SAME',
                          'NCHW', [4, 176, 8, 8])
        _test_convolution('conv_transpose', [4, 32, 8, 8], [2, 2, 176, 32], [1, 1], [1, 1], 'SAME',
                          'NCHW', [4, 176, 8, 8])
        _test_convolution('conv_transpose', [4, 32, 8, 8], [2, 2, 176, 32], [1, 1], [2, 2], 'SAME',
                          'NCHW', [4, 176, 15, 15])
        _test_convolution('conv_transpose', [4, 32, 8, 8], [3, 3, 176, 32], [1, 1], [1, 1], 'SAME',
                          'NCHW', [4, 176, 8, 8])
        _test_convolution('conv_transpose', [4, 32, 8, 8], [3, 3, 176, 32], [1, 1], [2, 2], 'SAME',
                          'NCHW', [4, 176, 15, 15])
        _test_convolution('conv_transpose', [4, 32, 8, 8], [3, 3, 176, 32], [1, 1], [2, 2], 'SAME',
                          'NCHW', [4, 176, 16, 16])
        _test_convolution('conv_transpose', [4, 19, 8, 8], [3, 3, 19, 19], [1, 1], [2, 2], 'VALID',
                          'NCHW', [4, 19, 17, 17])
        _test_convolution('conv_transpose', [4, 19, 17, 17], [1, 1, 124, 19], [1, 1], [1, 1], 'SAME',
                          'NCHW', [4, 124, 17, 17])
        _test_convolution('conv_transpose', [4, 19, 17, 17], [3, 3, 124, 19], [1, 1], [1, 1], 'SAME',
                          'NCHW', [4, 124, 17, 17])
        _test_convolution('conv_transpose', [4, 32, 8, 8], [3, 3, 12, 32], [1, 1], [2, 2], 'VALID',
                          'NCHW', [4, 12, 17, 17])
        # kernel 2x2, strides (2,2)
        _test_convolution('conv_transpose', [4, 19, 8, 8], [2, 2, 19, 19], [1, 1], [2, 2], 'VALID',
                          'NCHW', [4, 19, 16, 16])
        _test_convolution('conv_transpose', [4, 32, 8, 8], [2, 2, 12, 32], [1, 1], [2, 2], 'VALID',
                          'NCHW', [4, 12, 16, 16])
        # output channel is 1
        _test_convolution('conv_transpose', [1, 19, 8, 8], [1, 1, 1, 19], [1, 1], [1, 1], 'VALID',
                          'NCHW', [1, 1, 8, 8])
        """

    _test_convolution("conv", [4, 8, 8, 176], [1, 1, 176, 32], [1, 1], [1, 1], "SAME", "NHWC")
    _test_convolution("conv", [4, 17, 17, 19], [3, 3, 19, 19], [1, 1], [2, 2], "VALID", "NHWC")
    _test_convolution("conv", [4, 17, 17, 124], [1, 1, 124, 19], [1, 1], [1, 1], "SAME", "NHWC")
    _test_convolution("conv", [4, 17, 17, 12], [3, 3, 12, 32], [1, 1], [2, 2], "VALID", "NHWC")
    _test_convolution("depthwise", [4, 8, 8, 176], [1, 1, 176, 1], [1, 1], [1, 1], "SAME", "NHWC")
    _test_convolution("depthwise", [4, 17, 17, 19], [3, 3, 19, 1], [1, 1], [2, 2], "VALID", "NHWC")
    _test_convolution("depthwise", [4, 17, 17, 124], [1, 1, 124, 1], [1, 1], [1, 1], "SAME", "NHWC")
    _test_convolution("depthwise", [4, 17, 17, 12], [3, 3, 12, 1], [1, 1], [2, 2], "VALID", "NHWC")
    _test_convolution("depthwise", [4, 17, 17, 12], [3, 3, 12, 2], [1, 1], [2, 2], "VALID", "NHWC")
    """
    _test_convolution('conv_transpose', [4, 8, 8, 32], [1, 1, 176, 32], [1, 1], [1, 1], 'SAME',
                      'NHWC', [4, 8, 8, 176])
    _test_convolution('conv_transpose', [4, 8, 8, 32], [2, 2, 176, 32], [1, 1], [1, 1], 'SAME',
                      'NHWC', [4, 8, 8, 176])
    _test_convolution('conv_transpose', [4, 8, 8, 32], [2, 2, 176, 32], [1, 1], [2, 2], 'SAME',
                      'NHWC', [4, 15, 15, 176])
    _test_convolution('conv_transpose', [4, 8, 8, 32], [3, 3, 176, 32], [1, 1], [1, 1], 'SAME',
                      'NHWC', [4, 8, 8, 176])
    _test_convolution('conv_transpose', [4, 8, 8, 32], [3, 3, 176, 32], [1, 1], [2, 2], 'SAME',
                      'NHWC', [4, 15, 15, 176])
    _test_convolution('conv_transpose', [4, 8, 8, 32], [3, 3, 176, 32], [1, 1], [2, 2], 'SAME',
                      'NHWC', [4, 16, 16, 176])
    _test_convolution('conv_transpose', [4, 8, 8, 19], [3, 3, 19, 19], [1, 1], [2, 2], 'VALID',
                      'NHWC', [4, 17, 17, 19])
    _test_convolution('conv_transpose', [4, 17, 17, 19], [1, 1, 124, 19], [1, 1], [1, 1], 'SAME',
                      'NHWC', [4, 17, 17, 124])
    _test_convolution('conv_transpose', [4, 17, 17, 19], [3, 3, 124, 19], [1, 1], [1, 1], 'SAME',
                      'NHWC', [4, 17, 17, 124])
    _test_convolution('conv_transpose', [4, 8, 8, 32], [3, 3, 12, 32], [1, 1], [2, 2], 'VALID',
                      'NHWC', [4, 17, 17, 12])
    # kernel 2x2, strides (2,2)
    _test_convolution('conv_transpose', [4, 8, 8, 19], [2, 2, 19, 19], [1, 1], [2, 2], 'VALID',
                      'NHWC', [4, 16, 16, 19])
    _test_convolution('conv_transpose', [4, 8, 8, 32], [2, 2, 12, 32], [1, 1], [2, 2], 'VALID',
                      'NHWC', [4, 16, 16, 12])
    # output channel is 1
    _test_convolution('conv_transpose', [1, 8, 8, 19], [1, 1, 1, 19], [1, 1], [1, 1], 'VALID',
                      'NHWC', [1, 8, 8, 1])
    """


#######################################################################
# Convolution3D
# -------------


def _test_convolution3d(
    opname,
    tensor_in_sizes,
    filter_in_sizes,
    dilations,
    strides,
    padding,
    data_format,
    deconv_output_shape=[],
):
    """ One iteration of 3D convolution with given shapes and attributes """
    filter_data = tf.convert_to_tensor(
        np.random.uniform(-1, 1, size=filter_in_sizes).astype("float32")
    )
    if data_format == "NDHWC":
        strides = [1] + strides + [1]
        dilations = [1] + dilations + [1]
    else:
        strides = [1, 1] + strides
        dilations = [1, 1] + dilations

    root = tracking.AutoTrackable()
    root.f = def_function.function(
        lambda x: tf.nn.conv3d(
            x,
            filter_data,
            strides=strides,
            dilations=dilations,
            padding=padding,
            data_format=data_format,
        )
    )
    concrete_func = root.f.get_concrete_function(
        tensor_spec.TensorSpec(tensor_in_sizes, dtypes.float32)
    )

    input_data = tf.random.uniform(tensor_in_sizes, minval=-0.5, maxval=0.5, dtype=dtypes.float32)
    compare_tf_with_tvm_v2([input_data], concrete_func)


def test_forward_convolution3d():
    if is_gpu_available():
        _test_convolution3d(
            "conv", [4, 176, 8, 8, 8], [1, 1, 1, 176, 32], [1, 1, 1], [1, 1, 1], "SAME", "NCDHW"
        )
        _test_convolution3d(
            "conv", [4, 19, 17, 17, 17], [3, 3, 3, 19, 19], [1, 1, 1], [2, 2, 2], "VALID", "NCDHW"
        )
        _test_convolution3d(
            "conv", [4, 124, 17, 17, 17], [1, 1, 1, 124, 19], [1, 1, 1], [1, 1, 1], "SAME", "NCDHW"
        )
        _test_convolution3d(
            "conv", [4, 12, 17, 17, 17], [3, 3, 3, 12, 32], [1, 1, 1], [2, 2, 2], "VALID", "NCDHW"
        )
    """ TODO
    _test_convolution3d('conv', [4, 8, 8, 8, 176], [1, 1, 1, 176, 32], [1, 1, 1], [1, 1, 1], 'SAME', 'NDHWC')
    _test_convolution3d('conv', [4, 17, 17, 17, 19], [3, 3, 3, 19, 19], [1, 1, 1], [2, 2, 2], 'VALID', 'NDHWC')
    _test_convolution3d('conv', [4, 17, 17, 17, 124], [1, 1, 1, 124, 19], [1, 1, 1], [1, 1, 1], 'SAME', 'NDHWC')
    _test_convolution3d('conv', [4, 17, 17, 17, 12], [3, 3, 3, 12, 32], [1, 1, 1], [2, 2, 2], 'VALID', 'NDHWC')
    """


#######################################################################
# BiasAdd
# -------


def _test_biasadd(tensor_in_sizes, data_format):
    """ One iteration of biasadd with given shapes and attributes """

    tensor_bias_sizes = [tensor_in_sizes[1]] if data_format == "NCHW" else [tensor_in_sizes[3]]

    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x, y: tf.nn.bias_add(x, y, data_format=data_format))
    concrete_func = root.f.get_concrete_function(
        tensor_spec.TensorSpec(tensor_in_sizes, dtypes.float32),
        tensor_spec.TensorSpec(tensor_bias_sizes, dtypes.float32),
    )

    input_data = tf.random.uniform(tensor_in_sizes, dtype=dtypes.float32)
    bias_data = tf.random.uniform(tensor_bias_sizes, dtype=dtypes.float32)
    compare_tf_with_tvm_v2([input_data, bias_data], concrete_func)


def test_forward_biasadd():
    if is_gpu_available():
        _test_biasadd([4, 176, 8, 8], "NCHW")
        _test_biasadd([1, 100, 1, 1], "NCHW")
        _test_biasadd([4, 19, 17, 17], "NCHW")
        _test_biasadd([4, 124, 3, 3], "NCHW")

    _test_biasadd([4, 8, 8, 176], "NHWC")
    _test_biasadd([1, 1, 1, 100], "NHWC")
    _test_biasadd([4, 17, 17, 19], "NHWC")
    _test_biasadd([4, 3, 3, 124], "NHWC")


'''
def _test_forward_where(input_shape):
    with tf.Graph().as_default():
        dtype = tf.float32
        t = tf.constant(
            np.random.choice([0, 1, -2, 3, -1, 0.1, -0.2], size=input_shape).astype(dtype.name)
        )
        out = tf.where(t)
        compare_tf_with_tvm([], [], out.name, mode="debug")
        compare_tf_with_tvm([], [], out.name, mode="vm")


def test_forward_argwhere():
    _test_forward_where((5,))
    _test_forward_where((5, 5))
    _test_forward_where((5, 5, 5))
    _test_forward_where((5, 5, 5, 5))
    _test_forward_where((5, 5, 5, 5, 5))
'''


#######################################################################
# SpaceToBatchND
# --------------


def _test_space_to_batch_nd(input_shape, block_shape, paddings, dtype=dtypes.int32):
    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x: tf.space_to_batch_nd(x, block_shape, paddings))
    concrete_func = root.f.get_concrete_function(tensor_spec.TensorSpec(input_shape, dtype))

    input_data = tf.random.uniform(input_shape, minval=0, maxval=5, dtype=dtype)
    compare_tf_with_tvm_v2([input_data], concrete_func)


def test_forward_space_to_batch_nd():
    # test cases: https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/space-to-batch-n-d
    _test_space_to_batch_nd(input_shape=[1, 2, 2, 1], block_shape=[2, 2], paddings=[[0, 0], [0, 0]])

    _test_space_to_batch_nd(input_shape=[1, 2, 2, 3], block_shape=[2, 2], paddings=[[0, 0], [0, 0]])

    _test_space_to_batch_nd(input_shape=[1, 4, 4, 1], block_shape=[2, 2], paddings=[[0, 0], [0, 0]])

    _test_space_to_batch_nd(
        input_shape=[2, 2, 4, 1], block_shape=[2, 2], paddings=[[0, 0], [2, 0]], dtype=dtypes.int64
    )

    # pylint: disable=line-too-long
    # https://github.com/tensorflow/tensorflow/blob/24f578/tensorflow/python/kernel_tests/spacetobatch_op_test.py
    _test_space_to_batch_nd(input_shape=[2, 3], block_shape=[2], paddings=[[1, 0]], dtype="float32")

    _test_space_to_batch_nd(
        input_shape=[2, 3, 2], block_shape=[2], paddings=[[1, 0]], dtype="float64"
    )


#######################################################################
# BatchToSpaceND
# --------------


def _test_batch_to_space_nd(input_shape, block_shape, crops, dtype=dtypes.int32):
    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x: tf.compat.v1.batch_to_space_nd(x, block_shape, crops))
    concrete_func = root.f.get_concrete_function(tensor_spec.TensorSpec(input_shape, dtype))

    input_data = tf.random.uniform(input_shape, minval=0, maxval=5, dtype=dtype)
    compare_tf_with_tvm_v2([input_data], concrete_func)


def test_forward_batch_to_space_nd():
    # test cases: https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/batch-to-space-n-d
    _test_batch_to_space_nd(input_shape=[4, 1, 1, 1], block_shape=[2, 2], crops=[[0, 0], [0, 0]])

    _test_batch_to_space_nd(input_shape=[4, 1, 1, 3], block_shape=[2, 2], crops=[[0, 0], [0, 0]])

    _test_batch_to_space_nd(input_shape=[4, 2, 2, 1], block_shape=[2, 2], crops=[[0, 0], [0, 0]])

    _test_batch_to_space_nd(
        input_shape=[8, 1, 3, 1], block_shape=[2, 2], crops=[[0, 0], [2, 0]], dtype=dtypes.int64
    )

    # pylint: disable=line-too-long
    # https://github.com/tensorflow/tensorflow/blob/24f578/tensorflow/python/kernel_tests/batchtospace_op_test.py
    _test_batch_to_space_nd(
        input_shape=[18, 2, 1, 2], block_shape=[2, 3], crops=[[1, 1], [0, 0]], dtype="float32"
    )

    _test_batch_to_space_nd(
        input_shape=[20, 5, 8, 7], block_shape=[2, 2], crops=[[1, 1], [1, 1]], dtype="float64"
    )


#######################################################################
# Reshape
# -------


def _test_reshape(data, out_shape):
    """ One iteration of reshape operation with given data and out shape """
    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x: tf.reshape(x, out_shape))
    concrete_func = root.f.get_concrete_function(tensor_spec.TensorSpec(data.shape, dtypes.float32))

    input_data = tf.convert_to_tensor(data, dtype=dtypes.float32)
    compare_tf_with_tvm_v2([input_data], concrete_func)


def _test_reshape_with_call():
    """ relay.expr.Call as shape """
    root = tracking.AutoTrackable()
    data = np.zeros((6, 4, 2))
    out_shape = tf.constant([1, 2, 3], dtype="int32")
    root.f = def_function.function(lambda x: tf.reshape(x, tf.multiply(out_shape, 2)))
    concrete_func = root.f.get_concrete_function(tensor_spec.TensorSpec(data.shape, dtypes.float32))

    input_data = tf.convert_to_tensor(data, dtype=dtypes.float32)
    compare_tf_with_tvm_v2([input_data], concrete_func)


def _test_reshape_like(data, shape_like):
    """ A special case for reshape. """
    root = tracking.AutoTrackable()
    root.f = def_function.function(
        lambda x: tf.reshape(x, tf.shape(tf.convert_to_tensor(shape_like, dtype=dtypes.float32)))
    )
    concrete_func = root.f.get_concrete_function(tensor_spec.TensorSpec(data.shape, dtypes.float32))

    input_data = tf.convert_to_tensor(data, dtype=dtypes.float32)
    compare_tf_with_tvm_v2([input_data], concrete_func)


def test_forward_reshape():
    _test_reshape(np.arange(6.0), [2, 3])
    _test_reshape(np.arange(6), [-1, 2])
    _test_reshape(np.arange(6), [3, -1])
    _test_reshape(np.arange(6), [-1])
    _test_reshape_with_call()
    _test_reshape_like(np.zeros((3, 6)), np.zeros((9, 2)))


#######################################################################
# DepthToSpace
# ------------


def _test_depthtospace(data, block_size):
    """ One iteration of depth_to_space operation with given data and block size """
    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x: tf.nn.depth_to_space(x, block_size))
    concrete_func = root.f.get_concrete_function(tensor_spec.TensorSpec(data.shape, dtypes.float32))

    input_data = tf.convert_to_tensor(data, dtype=dtypes.float32)
    compare_tf_with_tvm_v2([input_data], concrete_func)


def test_forward_depthtospace():
    _test_depthtospace(np.random.normal(size=[1, 32, 32, 4]), 2)
    _test_depthtospace(np.random.normal(size=[1, 16, 8, 32]), 4)


#######################################################################
# SpaceToDepth
# ------------


def _test_spacetodepth(data, block_size):
    """ One iteration of space_to_depth operation with given data and block size """
    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x: tf.nn.space_to_depth(x, block_size))
    concrete_func = root.f.get_concrete_function(tensor_spec.TensorSpec(data.shape, dtypes.float32))

    input_data = tf.convert_to_tensor(data, dtype=dtypes.float32)
    compare_tf_with_tvm_v2([input_data], concrete_func)


def test_forward_spacetodepth():
    _test_spacetodepth(np.random.normal(size=[1, 32, 32, 4]), 2)
    _test_spacetodepth(np.random.normal(size=[1, 16, 8, 32]), 4)


#######################################################################
# Squeeze
# -------


def _test_squeeze(data, squeeze_dims=None):
    """ One iteration of squeeze """

    if squeeze_dims is None:
        squeeze_dims = []

    root = tracking.AutoTrackable()
    if squeeze_dims:
        root.f = def_function.function(lambda x: tf.squeeze(x, squeeze_dims))
    else:
        root.f = def_function.function(lambda x: tf.squeeze(x))

    concrete_func = root.f.get_concrete_function(tensor_spec.TensorSpec(data.shape, dtypes.float32))

    input_data = tf.convert_to_tensor(data, dtype=dtypes.float32)
    compare_tf_with_tvm_v2([input_data], concrete_func)


def test_forward_squeeze():
    """ Squeeze """

    # Nothing to squeeze.
    _test_squeeze(np.arange(2).reshape((2)))
    _test_squeeze(np.arange(6).reshape((2, 3)))

    # Squeeze the middle element away.
    _test_squeeze(np.arange(4).reshape((2, 1, 2)))

    # Squeeze on both ends.
    _test_squeeze(np.arange(6).reshape((1, 2, 1, 3, 1)))

    # Positive squeeze dim index.
    _test_squeeze(np.arange(6).reshape((1, 2, 1, 3, 1)), [0])
    _test_squeeze(np.arange(6).reshape((1, 2, 1, 3, 1)), [2, 4])
    _test_squeeze(np.arange(6).reshape((1, 2, 1, 3, 1)), [0, 4, 2])

    # Negative squeeze dim index.
    _test_squeeze(np.arange(6).reshape((1, 2, 1, 3, 1)), [-1])
    _test_squeeze(np.arange(6).reshape((1, 2, 1, 3, 1)), [-3, -5])
    _test_squeeze(np.arange(6).reshape((1, 2, 1, 3, 1)), [-3, -5, -1])


'''
# TODO: The following operators are not implemented: {'TensorListGetItem', 'TensorListSetItem', 'TensorListReserve'}
def test_tensor_array_constructor():
    def run(dtype_str):
        dtype = tf_dtypes[dtype_str]
        t = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]]).astype(dtype_str), dtype=dtype)
        t2 = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]]).astype(dtype_str), dtype=dtype)

        root = tracking.AutoTrackable()
        root.f = def_function.function(
            lambda: tf.TensorArray(dtype=dtype, size=2, infer_shape=False, dynamic_size=False)
            .write(0, t)
            .write(1, t2)
            .read(0)
        )
        concrete_func = root.f.get_concrete_function()
        compare_tf_with_tvm_v2([], concrete_func)

    for dtype in tf_dtypes.keys():
        run(dtype)


# TODO The following operators are not implemented: {'TensorListReserve', 'TensorListScatterIntoExistingList', 'TensorListGetItem'}
def test_tensor_array_scatter():
    def run(dtype_str):
        dtype = tf_dtypes[dtype_str]
        t = tf.constant(np.array([[1.0], [2.0], [3.0]]).astype(dtype_str), dtype=dtype)
        indices = tf.constant([2, 1, 0])

        root = tracking.AutoTrackable()

        @def_function.function
        def array_func():
            ta1 = tf.TensorArray(dtype=dtype, size=3, infer_shape=False, dynamic_size=False)
            ta2 = ta1.scatter(indices, t)
            out0 = ta2.read(0)
            out1 = ta2.read(1)
            out2 = ta2.read(2)
            return out0, out1, out2

        root.f = def_function.function = array_func
        concrete_func = root.f.get_concrete_function()
        compare_tf_with_tvm_v2([], concrete_func)

    for dtype in tf_dtypes.keys():
        run(dtype)


# TODO(wweic): Fix gather issue with PartialEvaluate
# def test_tensor_array_gather():
#     with tf.Graph().as_default():
#         dtype = 'float32'
#         t = tf.constant([[1.0], [2.0], [3.0]])
#         scatter_indices = tf.constant([2, 1, 0])
#         gather_indices = tf.constant([1, 2])
#         ta1 = tf.TensorArray(dtype=tf.float32, size=3, infer_shape=False, dynamic_size=False)
#         ta2 = ta1.scatter(scatter_indices, t)
#         t1 = ta2.gather(gather_indices)
#         g = tf.get_default_graph()
#         compare_tf_with_tvm([], [], ['TensorArrayGatherV3:0'], mode='debug')

# TODO: The following operators are not implemented: {'TensorListGetItem', 'TensorListSplit'}
def test_tensor_array_split():
    def run(dtype_str):
        dtype = tf_dtypes[dtype_str]
        t = tf.constant(
            np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]]).astype(dtype_str),
            dtype=dtype,
        )
        split_length = tf.constant([2, 2, 2, 2], dtype=tf.int32)

        root = tracking.AutoTrackable()

        @def_function.function
        def array_func():
            ta1 = tf.TensorArray(dtype=dtype, size=4, infer_shape=False, dynamic_size=False)
            ta2 = ta1.split(t, split_length)
            out0 = ta2.read(0)
            out1 = ta2.read(1)
            out2 = ta2.read(2)
            out3 = ta2.read(3)
            return out0, out1, out2, out3

        root.f = def_function.function = array_func
        concrete_func = root.f.get_concrete_function()
        compare_tf_with_tvm_v2([], concrete_func)

    for dtype in tf_dtypes.keys():
        run(dtype)


# TODO: The following operators are not implemented: {'TensorListSplit', 'TensorListConcatV2'}
def test_tensor_array_concat():
    def run(dtype_str):
        dtype = tf_dtypes[dtype_str]
        t = tf.constant(
            np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]]).astype(dtype_str),
            dtype=dtype,
        )
        split_length = tf.constant([2, 2, 2, 2], dtype=tf.int32)

        root = tracking.AutoTrackable()

        @def_function.function
        def array_func():
            ta1 = tf.TensorArray(dtype=dtype, size=4, infer_shape=False, dynamic_size=False)
            ta2 = ta1.split(t, split_length)
            ta3 = ta2.concat()
            out = tf.identity(ta3)
            return out

        root.f = def_function.function = array_func
        concrete_func = root.f.get_concrete_function()
        compare_tf_with_tvm_v2([], concrete_func)

    for dtype in tf_dtypes.keys():
        run(dtype)


# TODO: The following operators are not implemented: {'TensorListReserve', 'TensorListLength'}
def test_tensor_array_size():
    def run(dtype_str):
        root = tracking.AutoTrackable()

        @def_function.function
        def array_func():
            ta1 = tf.TensorArray(dtype=dtype, size=2, infer_shape=False, dynamic_size=False)
            out = ta1.size()
            return out

        root.f = def_function.function = array_func
        concrete_func = root.f.get_concrete_function()
        compare_tf_with_tvm_v2([], concrete_func)

    for dtype in tf_dtypes.keys():
        run(dtype)


# TODO : The following operators are not implemented: {'TensorListLength', 'TensorListGetItem', 'TensorListFromTensor'}
def test_tensor_array_unstack():
    def run(dtype_str, input_shape):
        dtype = tf_dtypes[dtype_str]
        t = tf.constant(np.random.choice([0, 1, 2, 3], size=input_shape).astype(dtype.name))
        root = tracking.AutoTrackable()

        @def_function.function
        def array_func():
            ta1 = tf.TensorArray(dtype=dtype, infer_shape=False, size=input_shape[0])
            ta2 = ta1.unstack(t)
            out0 = ta2.size()
            out1 = ta2.read(0)
            return out0, out1

        root.f = def_function.function = array_func
        concrete_func = root.f.get_concrete_function()
        compare_tf_with_tvm_v2([], concrete_func)

    for dtype in tf_dtypes.keys():
        run(dtype, (5,))
        run(dtype, (5, 5))
        run(dtype, (5, 5, 5))
        run(dtype, (5, 5, 5, 5))
        run(dtype, (5, 5, 5, 5, 5))
        run(dtype, (5, 5, 5, 5, 5, 5))
'''


#######################################################################
# Reductions
# ----------


def test_reductions():
    api_dict = {
        tf.math.argmax: [
            {
                "tensor_args": [
                    {"min_val": -100, "max_val": 100, "dtype": dtypes.float32, "shape": (8, 4, 9)},
                ],
                "ext_args": {"axis": None, "output_type": dtypes.int32},
            },
            {
                "tensor_args": [
                    {"min_val": -100, "max_val": 100, "dtype": dtypes.float32, "shape": (8, 4, 9)},
                ],
                "ext_args": {"axis": 0, "output_type": dtypes.int32},
            },
            {
                "tensor_args": [
                    {"min_val": -100, "max_val": 100, "dtype": dtypes.float32, "shape": (8, 4, 9)},
                ],
                "ext_args": {"axis": 1, "output_type": dtypes.int32},
            },
            {
                "tensor_args": [
                    {"min_val": -100, "max_val": 100, "dtype": dtypes.float32, "shape": (8, 4, 9)},
                ],
                "ext_args": {"axis": 2, "output_type": dtypes.int32},
            },
        ],
        tf.math.argmin: [
            {
                "tensor_args": [
                    {"min_val": -100, "max_val": 100, "dtype": dtypes.float32, "shape": (8, 4, 9)},
                ],
                "ext_args": {"axis": None, "output_type": dtypes.int32},
            },
            {
                "tensor_args": [
                    {"min_val": -100, "max_val": 100, "dtype": dtypes.float32, "shape": (8, 4, 9)},
                ],
                "ext_args": {"axis": 0, "output_type": dtypes.int32},
            },
            {
                "tensor_args": [
                    {"min_val": -100, "max_val": 100, "dtype": dtypes.float32, "shape": (8, 4, 9)},
                ],
                "ext_args": {"axis": 1, "output_type": dtypes.int32},
            },
            {
                "tensor_args": [
                    {"min_val": -100, "max_val": 100, "dtype": dtypes.float32, "shape": (8, 4, 9)},
                ],
                "ext_args": {"axis": 2, "output_type": dtypes.int32},
            },
        ],
        tf.reduce_sum: [
            {
                "tensor_args": [
                    {"min_val": -100, "max_val": 100, "dtype": dtypes.float32, "shape": (8, 4, 9)},
                ],
                "ext_args": {},
            },
            {
                "tensor_args": [
                    {"min_val": -100, "max_val": 100, "dtype": dtypes.float32, "shape": (8, 4, 9)},
                ],
                "ext_args": {"axis": 0},
            },
            {
                "tensor_args": [
                    {"min_val": -100, "max_val": 100, "dtype": dtypes.float32, "shape": (8, 4, 9)},
                ],
                "ext_args": {"axis": (0, 1)},
            },
        ],
        tf.keras.backend.mean: [
            {
                "tensor_args": [
                    {
                        "min_val": -100,
                        "max_val": 100,
                        "dtype": dtypes.float32,
                        "shape": (10, 8, 16, 32),
                    },
                ],
                "ext_args": {},
            },
            {
                "tensor_args": [
                    {
                        "min_val": -100,
                        "max_val": 100,
                        "dtype": dtypes.float32,
                        "shape": (10, 8, 16, 32),
                    },
                ],
                "ext_args": {"axis": (2, 3)},
            },
            {
                "tensor_args": [
                    {
                        "min_val": -100,
                        "max_val": 100,
                        "dtype": dtypes.float32,
                        "shape": (10, 8, 16, 32),
                    },
                ],
                "ext_args": {"axis": (2, 3), "keepdims": True},
            },
        ],
        tf.math.reduce_prod: [
            {
                "tensor_args": [
                    {"min_val": -100, "max_val": 100, "dtype": dtypes.float32, "shape": (5,)},
                ],
                "ext_args": {"axis": 0, "keepdims": False},
            },
            {
                "tensor_args": [
                    {"min_val": -100, "max_val": 100, "dtype": dtypes.float32, "shape": (5, 5)},
                ],
                "ext_args": {"axis": 0, "keepdims": False},
            },
            {
                "tensor_args": [
                    {"min_val": -100, "max_val": 100, "dtype": dtypes.float32, "shape": (5, 5)},
                ],
                "ext_args": {"axis": 1, "keepdims": False},
            },
            {
                "tensor_args": [
                    {"min_val": -100, "max_val": 100, "dtype": dtypes.float32, "shape": (5,)},
                ],
                "ext_args": {"axis": 0, "keepdims": True},
            },
            {
                "tensor_args": [
                    {"min_val": -100, "max_val": 100, "dtype": dtypes.float32, "shape": (5, 5)},
                ],
                "ext_args": {"axis": 0, "keepdims": True},
            },
            {
                "tensor_args": [
                    {"min_val": -100, "max_val": 100, "dtype": dtypes.float32, "shape": (5, 5)},
                ],
                "ext_args": {"axis": 1, "keepdims": True},
            },
        ],
    }
    api_dict = {
        tf.math.reduce_max: [
            {
                "tensor_args": [
                    {
                        "min_val": -100,
                        "max_val": 100,
                        "dtype": dtypes.int32,
                        "shape": (10, 8, 16, 32),
                    },
                ],
                "ext_args": {"axis": (-1,), "keepdims": True},
            },
            {
                "tensor_args": [
                    {
                        "min_val": -100,
                        "max_val": 100,
                        "dtype": dtypes.float32,
                        "shape": (10, 8, 16, 32),
                    },
                ],
                "ext_args": {"axis": (2, 3), "keepdims": True},
            },
            {
                "tensor_args": [
                    {
                        "min_val": -100,
                        "max_val": 100,
                        "dtype": dtypes.float32,
                        "shape": (10, 8, 16, 32),
                    },
                ],
                "ext_args": {"axis": (1, 2), "keepdims": True},
            },
        ],
        tf.math.reduce_min: [
            {
                "tensor_args": [
                    {
                        "min_val": -100,
                        "max_val": 100,
                        "dtype": dtypes.int32,
                        "shape": (10, 8, 16, 32),
                    },
                ],
                "ext_args": {"axis": (-1,), "keepdims": True},
            },
            {
                "tensor_args": [
                    {
                        "min_val": -100,
                        "max_val": 100,
                        "dtype": dtypes.float32,
                        "shape": (10, 8, 16, 32),
                    },
                ],
                "ext_args": {"axis": (2, 3), "keepdims": True},
            },
            {
                "tensor_args": [
                    {
                        "min_val": -100,
                        "max_val": 100,
                        "dtype": dtypes.float32,
                        "shape": (10, 8, 16, 32),
                    },
                ],
                "ext_args": {"axis": (1, 2), "keepdims": True},
            },
        ],
    }

    for api, params in api_dict.items():
        for param in params:
            _test_forward_op(api, param)


#######################################################################
# All, Any
# --------


def test_forward_reduce_all():
    """Test the All operator."""
    np_data = np.random.choice([True, False], size=(5, 7, 11))
    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x: tf.reduce_all(x))
    concrete_func = root.f.get_concrete_function(
        tensor_spec.TensorSpec(np_data.shape, np_data.dtype)
    )

    input_data = tf.convert_to_tensor(np_data)
    compare_tf_with_tvm_v2([input_data], concrete_func)


def test_forward_reduce_any():
    """Test the Any operator."""
    np_data = np.random.choice([True, False], size=(5, 7, 11))
    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x: tf.reduce_any(x))
    concrete_func = root.f.get_concrete_function(
        tensor_spec.TensorSpec(np_data.shape, np_data.dtype)
    )

    input_data = tf.convert_to_tensor(np_data)
    compare_tf_with_tvm_v2([input_data], concrete_func)


#######################################################################
# MatMul, BatchMatMul, BatchMatMulV2
# ----------------------------------


def _test_matmul(i, j, k, dtype, outer=None):
    """ One iteration of matmul """

    A_shape_init = [i, j]
    B_shape_init = [j, k]

    for transpose_a in [False, True]:
        for transpose_b in [False, True]:
            outer = outer or []
            A_shape = outer + (A_shape_init[::-1] if transpose_a else A_shape_init)
            B_shape = outer + (B_shape_init[::-1] if transpose_b else B_shape_init)

            root = tracking.AutoTrackable()
            root.f = def_function.function(
                lambda x, y: tf.matmul(x, y, transpose_a=transpose_a, transpose_b=transpose_b)
            )
            concrete_func = root.f.get_concrete_function(
                tensor_spec.TensorSpec(A_shape, dtype), tensor_spec.TensorSpec(B_shape, dtype)
            )

            A_data = tf.random.uniform(A_shape, -10, 10, dtype)
            B_data = tf.random.uniform(B_shape, -10, 10, dtype)
            compare_tf_with_tvm_v2([A_data, B_data], concrete_func)


def test_forward_matmul():
    """ MatMul op test"""
    _test_matmul(1, 3, 6, dtypes.int32)
    _test_matmul(5, 3, 1, dtypes.float64)


def _test_batch_matmul(A_shape, B_shape, dtype, adjoint_a=False, adjoint_b=False):

    root = tracking.AutoTrackable()
    root.f = def_function.function(
        lambda x, y: tf.matmul(x, y, adjoint_a=adjoint_a, adjoint_b=adjoint_b)
    )
    concrete_func = root.f.get_concrete_function(
        tensor_spec.TensorSpec(A_shape, dtype), tensor_spec.TensorSpec(B_shape, dtype)
    )

    A_data = tf.random.uniform(A_shape, -1, 1, dtype)
    B_data = tf.random.uniform(B_shape, -1, 1, dtype)
    compare_tf_with_tvm_v2([A_data, B_data], concrete_func)


def test_forward_batch_matmul():
    """ TF op BatchMatMul, BatchMatMulV2 test"""
    _test_batch_matmul((3, 5, 4), (3, 4, 5), dtypes.int32)
    _test_batch_matmul((3, 5, 4), (3, 4, 5), dtypes.float32, True, True)
    _test_batch_matmul((3, 5, 4), (3, 5, 4), dtypes.int32, True, False)
    _test_batch_matmul((3, 5, 4), (3, 5, 4), dtypes.float32, False, True)
    _test_batch_matmul((2, 3, 4, 5, 6), (2, 3, 4, 6, 5), dtypes.int32)
    _test_batch_matmul((1, 2, 3, 4, 5, 6), (1, 2, 3, 4, 6, 5), dtypes.float32, True, True)
    _test_batch_matmul((3, 4, 5, 6), (3, 4, 5, 6), dtypes.int32, True, False)
    _test_batch_matmul(
        (2, 3, 4, 2, 3, 4, 5, 6), (2, 3, 4, 2, 3, 4, 5, 6), dtypes.float32, False, True
    )


#######################################################################
# StridedSlice
# ------------


def _test_stridedslice(
    ip_shape,
    begin,
    end,
    stride,
    dtype,
    begin_mask=0,
    end_mask=0,
    new_axis_mask=0,
    shrink_axis_mask=0,
    ellipsis_mask=0,
):
    """ One iteration of a Stridedslice """
    root = tracking.AutoTrackable()
    root.f = def_function.function(
        lambda x: tf.strided_slice(
            x,
            begin,
            end,
            stride,
            begin_mask=begin_mask,
            end_mask=end_mask,
            ellipsis_mask=ellipsis_mask,
            new_axis_mask=new_axis_mask,
            shrink_axis_mask=shrink_axis_mask,
            name="strided_slice",
        )
    )
    concrete_func = root.f.get_concrete_function(tensor_spec.TensorSpec(ip_shape, dtypes.float32))

    print("Shape:", ip_shape)
    input_data = tf.random.uniform(ip_shape, dtype=dtypes.float32)
    compare_tf_with_tvm_v2([input_data], concrete_func)

    """
    tf.reset_default_graph()
    in_data = tf.placeholder(dtype, ip_shape, name="in_data")
    tf.strided_slice(in_data, begin, end, stride, begin_mask=begin_mask,
                     end_mask=end_mask, new_axis_mask=new_axis_mask,
                     shrink_axis_mask=shrink_axis_mask,
                     ellipsis_mask=ellipsis_mask, name="strided_slice")
    np_data = np.random.uniform(size=ip_shape).astype(dtype)

    compare_tf_with_tvm(np_data, 'in_data:0', 'strided_slice:0')
    """


def test_forward_stridedslice():
    """test StridedSlice"""

    _test_stridedslice((2,), [1], [1], [1], "float32", shrink_axis_mask=1)
    _test_stridedslice((3, 4, 3), [1, -1, 0], [4, -5, 3], [2, -1, 1], "float32")
    _test_stridedslice((3, 4, 3), [1, 0], [4, 3], [2, 1], "float32", ellipsis_mask=8)
    _test_stridedslice((3, 4, 3), [1, 0], [4, 2], [2, 1], "float32", ellipsis_mask=2)
    _test_stridedslice((3, 4, 5, 3), [1, 0], [4, 2], [2, 1], "float32", ellipsis_mask=2)
    _test_stridedslice((3, 4, 5, 3), [1, 0, 1], [4, 2, 2], [2, 1, 1], "float32", ellipsis_mask=2)
    _test_stridedslice((3, 4, 3), [1, 1, 0], [4, 4, 2], [2, 1, 1], "float32", new_axis_mask=5)
    _test_stridedslice(
        (3, 4, 3), [1, 1, 1], [4, 4, 1], [2, 1, 1], "float32", ellipsis_mask=2, new_axis_mask=4
    )
    _test_stridedslice(
        (6, 4, 5), [1, 1, 1], [6, 3, 4], [2, 1, 1], "float32", ellipsis_mask=2, new_axis_mask=5
    )
    _test_stridedslice(
        (3, 4, 3), [1, 1, 2], [4, 4, 3], [2, 1, 1], "float32", ellipsis_mask=4, new_axis_mask=2
    )
    _test_stridedslice(
        (3, 4, 3), [1, 1, 2], [4, 4, 3], [2, 1, 1], "float32", ellipsis_mask=2, new_axis_mask=3
    )
    _test_stridedslice(
        (3, 4, 3), [1, 1, 0], [4, 4, 1], [2, 1, 1], "float32", ellipsis_mask=2, new_axis_mask=3
    )
    _test_stridedslice(
        (3, 4, 3), [1, 1, 2], [4, 4, 3], [2, 1, 1], "float32", ellipsis_mask=2, new_axis_mask=2
    )
    _test_stridedslice((3, 4), [1, 0], [4, 4], [1, 1], "float32", shrink_axis_mask=2)
    _test_stridedslice(
        (3, 4, 3), [1, 1, 0], [4, 4, 3], [2, 1, 1], "float32", shrink_axis_mask=2, new_axis_mask=2
    )
    _test_stridedslice(
        (3, 4, 3), [1, 1, 0], [4, 4, 3], [2, 1, 1], "float32", shrink_axis_mask=1, new_axis_mask=2
    )
    _test_stridedslice(
        (3, 4, 3), [1, 1, 0], [4, 4, 3], [2, 1, 1], "float32", shrink_axis_mask=2, new_axis_mask=1
    )
    _test_stridedslice(
        (3, 4, 5, 4, 5, 6), [0, 0], [2, 3], [1, 1], "float32", shrink_axis_mask=5, new_axis_mask=1
    )
    _test_stridedslice(
        (3, 4, 5, 4, 5, 6),
        [0, 0, 1, 2, 1],
        [2, 3, 4, 5, 3],
        [1, 1, 2, 2, 1],
        "float32",
        shrink_axis_mask=5,
        new_axis_mask=1,
        ellipsis_mask=2,
        begin_mask=8,
        end_mask=8,
    )
    _test_stridedslice(
        (3, 4, 5, 4, 5, 6),
        [0, 0, 1, 2, 1],
        [2, 3, 4, 5, 3],
        [1, 1, 2, 2, 1],
        "float32",
        shrink_axis_mask=8,
        new_axis_mask=1,
        ellipsis_mask=2,
        begin_mask=5,
        end_mask=5,
    )
    _test_stridedslice(
        (3, 4, 5, 4, 5, 6),
        [0, 0, 1, 2, 1],
        [2, 3, 4, 5, 3],
        [1, 1, 2, 2, 1],
        "float32",
        shrink_axis_mask=16,
        new_axis_mask=1,
        ellipsis_mask=2,
        begin_mask=5,
        end_mask=5,
    )
    _test_stridedslice(
        (3, 4, 5, 4, 5, 6),
        [1, 2, 0, -3],
        [4, 5, 3, 3],
        [2, 2, 1, 1],
        "float32",
        shrink_axis_mask=8,
        new_axis_mask=1,
        ellipsis_mask=2,
        begin_mask=5,
        end_mask=8,
    )


#######################################################################
# TruncateMod
# -----------
def _test_forward_truncatemod(ip_shape, dtype):
    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x, y: tf.truncatemod(x, y))
    concrete_func = root.f.get_concrete_function(
        tensor_spec.TensorSpec(ip_shape, dtype), tensor_spec.TensorSpec(ip_shape, dtype)
    )

    input_data_1 = tf.random.uniform(ip_shape, -100, 100, dtype=dtype)
    input_data_2 = tf.random.uniform(ip_shape, 1, 10, dtype=dtype)
    compare_tf_with_tvm_v2([input_data_1, input_data_2], concrete_func)


def test_forward_truncatemod():
    """test TruncateMod"""
    _test_forward_truncatemod((4, 3, 7), dtypes.int32)


#######################################################################
# Gather, GatherV2, GatherNd
# --------------------------


def _test_gather(ip_shape, indice_shape, indice_value, axis, dtype):
    """ One iteration of a GatherV2 """
    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x, y: tf.gather(x, y, axis=axis))
    concrete_func = root.f.get_concrete_function(
        tensor_spec.TensorSpec(ip_shape, dtypes.float32),
        tensor_spec.TensorSpec(indice_shape, dtypes.int32),
    )

    def _fill_indices(indice_value):
        indices = np.array(ip_shape, dtype=dtype)
        if isinstance(indice_value, int):
            indices = np.array([indice_value], dtype="int32")
        else:
            indices = np.asarray(indice_value, dtype="int32")
        return indices

    np_indices = _fill_indices(indice_value)

    input_data = tf.random.uniform(ip_shape, dtype=dtypes.float32)
    indice_data = tf.convert_to_tensor(np_indices)
    compare_tf_with_tvm_v2([input_data, indice_data], concrete_func)


def test_forward_gather():
    """test Gather/GatherV2 layer"""
    _test_gather((4,), (1,), 1, 0, "int32")
    _test_gather((4,), (1,), 1, 0, "float32")
    _test_gather((1, 4), (1,), [0], 0, "int32")
    _test_gather((4,), (1, 2, 2), [[[1, 0], [0, 1]]], 0, "float32")
    _test_gather((2, 2), (1, 2, 2), [[[1, 0], [0, 1]]], 0, "int32")
    _test_gather((2, 2), (1, 2, 2), [[[1, 0], [0, 1]]], 1, "int32")
    _test_gather((2, 2), (1, 2, 2), [[[1, 0], [0, 1]]], 0, "float32")
    _test_gather((3, 3, 3), (1, 1, 2), [[[1, 0]]], 0, "int32")
    _test_gather((3, 3, 3), (1, 1, 2), [[[1, 0]]], 2, "int32")
    _test_gather((4, 3, 5, 6), (1, 4), [[2, 1, 0, 0]], 0, "float32")


def test_forward_gather_nd():
    """test operator GatherNd"""
    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x: tf.gather(x, indices=[[1, 0], [0, 1]]))
    concrete_func = root.f.get_concrete_function(tensor_spec.TensorSpec((2, 2), dtypes.float32))

    input_data = tf.random.uniform((2, 2), 1, 100, dtype=dtypes.float32)
    compare_tf_with_tvm_v2([input_data], concrete_func)


'''
#######################################################################
# BiasAdd
# -------
def test_forward_bias_add():
    """test Op BiasAdd"""

    def check_bias_add(lh_shpae, rh_shape, dtype):
        tf.reset_default_graph()
        lh_data = np.random.uniform(size=lh_shpae).astype(dtype)
        rh_data = np.random.uniform(size=rh_shape).astype(dtype)
        lft_data = tf.placeholder(dtype, name="lft_data")
        rgt_data = tf.placeholder(dtype, name="rgt_data")
        tf.nn.bias_add(lft_data, rgt_data, name="BiasAdd")
        compare_tf_with_tvm([lh_data, rh_data], ["lft_data:0", "rgt_data:0"], "BiasAdd:0")

    check_bias_add((10, 8, 16, 32), (32,), dtype="int32")
    check_bias_add((10, 20), (20,), dtype="float32")
'''


#######################################################################
# Split
# -----


def _test_split(in_shape, axis, num_or_size_splits, dtype):
    """ One iteration of a Split """
    num_split = (
        len(num_or_size_splits) if isinstance(num_or_size_splits, list) else num_or_size_splits
    )
    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x: tf.split(x, num_or_size_splits, axis=axis))
    concrete_func = root.f.get_concrete_function(tensor_spec.TensorSpec(in_shape, dtypes.float32))

    input_data = tf.random.uniform(in_shape, dtype=dtypes.float32)
    compare_tf_with_tvm_v2([input_data], concrete_func)

    root = tracking.AutoTrackable()
    root.f = def_function.function(
        lambda x: tf.concat(tf.split(x, num_or_size_splits, axis=axis), axis=axis)
    )
    concrete_func = root.f.get_concrete_function(tensor_spec.TensorSpec(in_shape, dtypes.float32))

    compare_tf_with_tvm_v2([input_data], concrete_func)

    """
    np_data = np.random.uniform(-5, 5, size=in_shape).astype(dtype)
    tf.reset_default_graph()
    in_data = tf.placeholder(dtype, in_shape, name="in_data")
    num_split = len(num_or_size_splits) if isinstance(num_or_size_splits, list)\
        else num_or_size_splits
    split = tf.split(in_data, num_or_size_splits, axis=axis)
    relu = [tf.nn.relu(i) for i in split]

    compare_tf_with_tvm([np_data], ['in_data:0'], [n.name for n in relu])

    # and now test together with concat
    tf.reset_default_graph()
    in_data = tf.placeholder(dtype, in_shape, name="in_data")
    splitted = tf.split(in_data, num_or_size_splits, axis=axis)
    tf.concat(splitted, axis)

    compare_tf_with_tvm([np_data], 'in_data:0', 'concat:0')
    """


def test_forward_split():
    """test split layer"""
    # rank 1
    _test_split((3,), 0, 1, "float32")
    _test_split((3,), 0, 3, "float32")
    _test_split((6,), 0, 3, "float32")
    # rank 2
    _test_split((6, 2), 0, 3, "float32")
    _test_split((2, 6), 1, 6, "float32")
    # rank 3
    _test_split((6, 2, 4), 0, 2, "int32")
    _test_split((2, 6, 4), 1, 3, "float32")
    _test_split((2, 4, 6), 2, 1, "float32")
    # rank 4
    _test_split((6, 1, 3, 5), 0, 3, "float32")
    _test_split((1, 6, 3, 5), 1, 3, "float32")
    _test_split((1, 3, 6, 5), 2, 3, "float32")
    _test_split((1, 3, 5, 6), 3, 3, "float32")
    # split along negative axis
    _test_split((6, 1, 3, 5), -4, 3, "float32")
    _test_split((1, 6, 3, 5), -3, 3, "float32")
    _test_split((1, 3, 6, 5), -2, 3, "float32")
    _test_split((1, 3, 5, 6), -1, 3, "float32")
    # size_splits list
    _test_split((6,), 0, [1, 2, 3], "int32")
    _test_split((3, 6, 4), -2, [1, 4, 1], "float32")


######################################################################
# TopKV2
# ------


def _test_forward_top_k_v2(in_shape, k):
    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x: tf.math.top_k(x, k))
    concrete_func = root.f.get_concrete_function(tensor_spec.TensorSpec(in_shape, dtypes.float32))

    input_data = tf.random.uniform(in_shape, -100, 100, dtypes.float32)
    tf_out = concrete_func(input_data)
    compare_tf_with_tvm_v2([input_data], concrete_func, tf_out=[tf_out.values, tf_out.indices])


def test_forward_top_k_v2():
    _test_forward_top_k_v2((3,), 1)
    _test_forward_top_k_v2((3,), 3)
    _test_forward_top_k_v2((3, 5, 7), 3)
    _test_forward_top_k_v2((3, 5, 7), 3)


#######################################################################
# Tile
# ----


def _test_tile(in_shape, multiples, dtype):
    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x: tf.tile(x, multiples=multiples))
    concrete_func = root.f.get_concrete_function(tensor_spec.TensorSpec(in_shape, dtype))

    input_data = tf.random.uniform(in_shape, -100, 100, dtype=dtype)
    compare_tf_with_tvm_v2([input_data], concrete_func)


def test_forward_tile():
    """test Tile"""
    _test_tile((2,), (3,), dtypes.int32)
    _test_tile((2, 2), (2, 3), dtypes.float32)
    _test_tile((2, 4, 6), (6, 7, 8), dtypes.float64)


#######################################################################
# Resize Bilinear, Nearest_Neighbor
# ---------------------------------


def _test_resize_bilinear(in_shape, to_shape, align_corners):
    """ One iteration of resize bilinear """
    shape_data = np.array(to_shape).astype("int32")
    shape_data = constant_op.constant(shape_data, shape=shape_data.shape, dtype=shape_data.dtype)

    root = tracking.AutoTrackable()
    root.f = def_function.function(
        lambda x: tf.image.resize(x, shape_data, method=tf.image.ResizeMethod.BILINEAR)
    )
    concrete_func = root.f.get_concrete_function(tensor_spec.TensorSpec(in_shape, dtypes.float32))

    input_data = tf.random.uniform(in_shape, dtype=dtypes.float32)
    compare_tf_with_tvm_v2([input_data], concrete_func, opt_level=1)


def _test_resize_bilinear_from_tensor(in_shape, align_corners):
    """One iteration of resize bilinear with non-constant output shape, requires
    value inference to get proper output shape."""
    to_shape = in_shape(in_data)[1:3]

    root = tracking.AutoTrackable()
    root.f = def_function.function(
        lambda x: tf.image.resize(x, to_shape, method=tf.image.ResizeMethod.BILINEAR)
    )
    concrete_func = root.f.get_concrete_function(tensor_spec.TensorSpec(in_shape, dtypes.float32))

    input_data = tf.random.uniform(in_shape, dtype=dtypes.float32)
    compare_tf_with_tvm_v2([input_data], concrete_func, opt_level=1)


def _test_resize_nearest_neighbor(in_shape, to_shape):
    """ One iteration of resize nearest neighbor """
    shape_data = np.array(to_shape).astype("int32")
    shape_data = constant_op.constant(shape_data, shape=shape_data.shape, dtype=shape_data.dtype)

    root = tracking.AutoTrackable()
    root.f = def_function.function(
        lambda x: tf.image.resize(x, shape_data, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    )
    concrete_func = root.f.get_concrete_function(tensor_spec.TensorSpec(in_shape, dtypes.float32))

    input_data = tf.random.uniform(in_shape, dtype=dtypes.float32)
    compare_tf_with_tvm_v2([input_data], concrete_func, opt_level=1)


def _test_resize_nearest_neighbor_dynamic_shape(in_shape, scale):
    """ One iteration of resize nearest neighbor for graph with dynamic input shape """
    to_shape = in_shape(in_data)[1:3]

    root = tracking.AutoTrackable()
    root.f = def_function.function(
        lambda x: tf.image.resize(x, to_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    )
    concrete_func = root.f.get_concrete_function(tensor_spec.TensorSpec(in_shape, dtypes.float32))

    input_data = tf.random.uniform(in_shape, dtype=dtypes.float32)
    compare_tf_with_tvm_v2([input_data], concrete_func, opt_level=1)


'''
def test_forward_resize():
    """ Resize Bilinear, Nearest_Neighbor """
    # TF default layout is NHWC
    _test_resize_bilinear((4, 32, 32, 3), [50, 50], False)
    _test_resize_bilinear((6, 32, 32, 3), [20, 20], True)
    _test_resize_bilinear_from_tensor((4, 32, 32, 3), False)
    _test_resize_bilinear_from_tensor((6, 50, 50, 3), True)
    _test_resize_nearest_neighbor((6, 32, 32, 3), [20, 20])
    _test_resize_nearest_neighbor_dynamic_shape((1, 16, 16, 3), scale=[2, 2])
'''


#######################################################################
# BroadcastTo
# -----------


def _test_broadcast_to(ishape, to_shape):
    """ One iteration of broadcast_to"""
    shape_data = np.array(to_shape).astype("int32")
    shape_data = constant_op.constant(shape_data, shape=shape_data.shape, dtype=shape_data.dtype)

    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x: tf.broadcast_to(x, shape_data))
    concrete_func = root.f.get_concrete_function(tensor_spec.TensorSpec(ishape, dtypes.float32))

    input_data = tf.random.uniform(ishape)
    compare_tf_with_tvm_v2([input_data], concrete_func)


def _test_broadcast_to_from_tensor(in_shape):
    """ One iteration of broadcast_to with unknown shape at graph build"""
    data = np.random.uniform(size=in_shape).astype("float32")

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=[None], dtype=data.dtype)

        shape_data = tf.multiply(tf.shape(in_data), 32)
        tf.broadcast_to(in_data, shape_data)

        compare_tf_with_tvm(
            data, "Placeholder:0", "BroadcastTo:0"
        )  # TODO: Why not passing second arg ?


def test_forward_broadcast_to():
    """ Broadcast to """

    _test_broadcast_to((4, 1, 32, 32), [4, 8, 32, 32])
    _test_broadcast_to((6, 32, 32, 1), [6, 32, 32, 16])
    # _test_broadcast_to_from_tensor((1)) - TODO


#######################################################################
# Fill
# ----
def _test_fill(in_shape):
    """ Use the fill op to create a tensor of ones with non-constant shape."""
    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda: tf.ones(in_shape))
    concrete_func = root.f.get_concrete_function()

    compare_tf_with_tvm_v2([], concrete_func, opt_level=1)


def _test_fill_from_tensor(in_shape):
    """Use the fill op to create a tensor of ones with non-constant shape.
    Some extra ops need to be added here to prevent the graph from
    being fully constant and folded away."""

    root = tracking.AutoTrackable()

    @def_function.function
    def fill_test(x):
        ones = tf.ones(x.shape, dtype=dtypes.float32)
        out = tf.add(x, tf.reduce_mean(ones))
        return out

    root.f = fill_test
    concrete_func = root.f.get_concrete_function(tensor_spec.TensorSpec(in_shape, dtypes.float32))

    input_data = tf.random.uniform(in_shape, dtype=dtypes.float32)
    compare_tf_with_tvm_v2([input_data], concrete_func)


def test_forward_fill():
    """ Fill """

    _test_fill((32))
    _test_fill((6, 32, 64, 64))
    _test_fill_from_tensor((6, 32, 64, 64))


#######################################################################
# Crop to bounding box
# --------------------


def _test_crop(in_shape, off_h, off_w, tar_h, tar_w):
    """ Crop to bounding box """
    root = tracking.AutoTrackable()
    root.f = def_function.function(
        lambda x: tf.image.crop_to_bounding_box(x, off_h, off_w, tar_h, tar_w)
    )
    concrete_func = root.f.get_concrete_function(tensor_spec.TensorSpec(in_shape, dtypes.float32))

    input_data = tf.random.uniform(in_shape, dtype=dtypes.float32)
    compare_tf_with_tvm_v2([input_data], concrete_func, opt_level=1)


def test_forward_crop():
    """ Crop to bounding box """
    _test_crop((1, 224, 224, 3), 20, 20, 120, 120)


#######################################################################
# CropAndResize
# -------------


def _test_forward_crop_and_resize(
    img_shape,
    boxes,
    box_idx,
    crop_size,
    extrapolation_value=0.0,
    method="bilinear",
    dtype="float32",
):
    root = tracking.AutoTrackable()
    root.f = def_function.function(
        lambda x: tf.image.crop_and_resize(
            x,
            boxes=boxes,
            box_indices=box_idx,
            crop_size=crop_size,
            method=method,
            extrapolation_value=extrapolation_value,
            name="crop_and_resize",
        )
    )
    concrete_func = root.f.get_concrete_function(tensor_spec.TensorSpec(img_shape, dtypes.float32))

    input_data = tf.random.uniform(img_shape, dtype=dtypes.float32)
    compare_tf_with_tvm_v2([input_data], concrete_func, opt_level=1)


def test_forward_crop_and_resize():
    """ CropAndResize """
    _test_forward_crop_and_resize([1, 6, 6, 3], [[0, 0, 1, 1]], [0], [3, 3])
    _test_forward_crop_and_resize([1, 6, 6, 3], [[0, 0, 1, 1]], [0], [3, 3], 0.2)
    _test_forward_crop_and_resize([1, 6, 6, 3], [[0, 0, 1, 1]], [0], [3, 3], 0.2, "nearest")
    _test_forward_crop_and_resize([1, 11, 11, 3], [[0.3, 0.3, 1, 1]], [0], [21, 21])
    _test_forward_crop_and_resize([1, 41, 41, 3], [[0.2, 0.4, 0.8, 0.8]], [0], [21, 11])
    _test_forward_crop_and_resize([1, 100, 100, 3], [[0, 0, 0.9, 0.9]], [0], [30, 30])
    _test_forward_crop_and_resize([1, 224, 224, 3], [[0.1, 0.2, 1, 1]], [0], [9, 9])
    _test_forward_crop_and_resize([1, 249, 249, 3], [[0, 0, 1, 1]], [0], [9, 9])
    _test_forward_crop_and_resize([1, 201, 301, 3], [[0.2, 0.3, 0.7, 0.8]], [0], [51, 51])
    _test_forward_crop_and_resize(
        img_shape=[10, 11, 11, 3],
        boxes=[[0, 0, 0.9, 0.9], [0.2, 0.2, 0.8, 0.8]],
        box_idx=[0, 1],
        crop_size=[5, 5],
    )
    _test_forward_crop_and_resize(
        img_shape=[20, 576, 576, 3],
        boxes=[[0, 0, 1, 1], [0, 0, 0.8, 0.8], [0.1, 0.2, 0.9, 1], [0.2, 0, 1, 1]],
        box_idx=[1, 0, 2, 3],
        crop_size=[24, 24],
        extrapolation_value=0.3,
    )
    _test_forward_crop_and_resize(
        img_shape=[20, 229, 229, 3],
        boxes=[[0, 0, 0.9, 0.9], [0.3, 0.3, 1, 1], [0.2, 0.1, 0.7, 0.8], [0, 0, 1, 1]],
        box_idx=[3, 0, 2, 1],
        crop_size=[58, 58],
        extrapolation_value=0.2,
        method="nearest",
    )


def _test_forward_lstm():
    """test LSTM block cell"""
    _test_lstm_cell(1, 2, 1, 0.5, "float32")


#######################################################################
# Pack
# ---
def _test_pack(axis, shape, **kwargs):
    a = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    b = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x, y: tf.stack([x, y], axis=axis))
    concrete_func = root.f.get_concrete_function(
        tensor_spec.TensorSpec(shape, dtypes.float32), tensor_spec.TensorSpec(shape, dtypes.float32)
    )

    input_a = tf.convert_to_tensor(a, dtype=dtypes.float32)
    input_b = tf.convert_to_tensor(b, dtype=dtypes.float32)
    compare_tf_with_tvm_v2([input_a, input_b], concrete_func)


def test_forward_pack():
    for axis in range(-3, 3):
        _test_pack(axis, [3, 2, 1])
    for axis in range(-1, 1):
        _test_pack(axis, [3])
    _test_pack(0, [])


#######################################################################
# Unstack / Unpack
# ----------------


def _test_unpack(in_shape, axis, dtype):
    """test operator Unstack"""
    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x: tf.unstack(x, axis=axis, name="Unpack"))
    concrete_func = root.f.get_concrete_function(tensor_spec.TensorSpec(in_shape, dtype))

    input_data = tf.random.uniform(in_shape, -100, 100, dtype)
    compare_tf_with_tvm_v2([input_data], concrete_func)

    # Unstack + stack
    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x: tf.stack(tf.unstack(x, axis=axis), axis=axis))
    concrete_func = root.f.get_concrete_function(tensor_spec.TensorSpec(in_shape, dtype))

    compare_tf_with_tvm_v2([input_data], concrete_func)


def test_forward_unpack():
    """test unpack layer"""
    _test_unpack((6,), 0, dtypes.int32)
    _test_unpack((2, 6), 1, dtypes.float64)
    _test_unpack((21, 23, 3), 2, dtypes.float32)
    # negative axis
    _test_unpack((1, 4), -1, dtypes.int32)
    _test_unpack((3, 6, 4), -2, dtypes.float32)


#######################################################################
# Range
# -----


def test_forward_range():
    """test operator Range"""
    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda: tf.range(1, 18, 3))
    concrete_func = root.f.get_concrete_function()
    compare_tf_with_tvm_v2([], concrete_func)

    """test type assignment for operator Range"""
    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda: tf.range(1, 256 + 1, 1, dtype=dtypes.float32))
    concrete_func = root.f.get_concrete_function()
    compare_tf_with_tvm_v2([], concrete_func)


#######################################################################
# Pad
# ---


def _test_pad(in_shape, paddings, mode, **kwargs):
    """ One iteration of pad operation with given shape"""
    pad_values = constant_op.constant(paddings)
    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x: tf.pad(x, paddings=pad_values, mode=mode, **kwargs))
    concrete_func = root.f.get_concrete_function(tensor_spec.TensorSpec(in_shape, dtypes.float32))

    input_data = tf.random.uniform(in_shape, dtype=dtypes.float32)
    compare_tf_with_tvm_v2([input_data], concrete_func)


def test_forward_pad():
    """ Pad """
    _test_pad((2, 3), [[1, 1], [2, 2]], mode="CONSTANT")
    _test_pad((2, 3), [[1, 1], [2, 2]], mode="CONSTANT", constant_values=1.0)
    _test_pad((2, 3), [[1, 1], [2, 2]], mode="SYMMETRIC")
    _test_pad((2, 3), [[1, 1], [2, 2]], mode="REFLECT")


#######################################################################
# Logical operators
# -----------------


def _test_logical(api):
    np_data1 = np.random.choice(a=[False, True], size=(1, 4, 4, 3)).astype("bool")
    np_data2 = np.random.choice(a=[False, True], size=(1, 4, 4, 3)).astype("bool")

    in_data1 = tf.convert_to_tensor(np_data1)
    in_data2 = tf.convert_to_tensor(np_data2)

    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x, y: api(x, y))
    concrete_func = root.f.get_concrete_function(
        tensor_spec.TensorSpec((1, 4, 4, 3), dtypes.bool),
        tensor_spec.TensorSpec((1, 4, 4, 3), dtypes.bool),
    )
    compare_tf_with_tvm_v2([in_data1, in_data2], concrete_func)


def _test_logical_not():
    np_data = np.random.choice(a=[False, True], size=(1, 4, 4, 3)).astype("bool")

    in_data = tf.convert_to_tensor(np_data)

    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x: tf.logical_not(x))
    concrete_func = root.f.get_concrete_function(tensor_spec.TensorSpec((1, 4, 4, 3), dtypes.bool))
    compare_tf_with_tvm_v2([in_data], concrete_func)


def test_forward_logical():
    _test_logical(tf.logical_and)
    _test_logical(tf.logical_or)
    _test_logical(tf.math.logical_xor)
    _test_logical_not()


#######################################################################
# Where, Select
# -------------
def test_forward_where():
    """ Where: return elements depending on conditions"""
    in_shape, dtype = (1, 4, 4, 3), dtypes.float32
    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x, y: tf.where(x > y, x, y))
    concrete_func = root.f.get_concrete_function(
        tensor_spec.TensorSpec(in_shape, dtype), tensor_spec.TensorSpec(in_shape, dtype)
    )

    input_data_1 = tf.random.uniform(in_shape, dtype=dtype)
    input_data_2 = tf.random.uniform(in_shape, dtype=dtype)
    compare_tf_with_tvm_v2([input_data_1, input_data_2], concrete_func)


#######################################################################
# Tensorflow E2E Models from Tensorflow Hub
# -----------------------------------------


def test_forward_image_classification_models():
    """Test E2E models from Tensorflow Hub"""

    import tensorflow_hub as hub

    models = [
        # Inception V1
        {
            "url": "https://tfhub.dev/google/imagenet/inception_v1/classification/3",
            "tensor_arg": {
                "min_val": 0,
                "max_val": 1,
                "dtype": dtypes.float32,
                "shape": (1, 224, 224, 3),
            },
        },
        # Inception V3
        {
            "url": "https://tfhub.dev/google/imagenet/inception_v3/classification/3",
            "tensor_arg": {
                "min_val": 0,
                "max_val": 1,
                "dtype": dtypes.float32,
                "shape": (1, 224, 224, 3),
            },
        },
        # MobileNet V1
        {
            "url": "https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/3",
            "tensor_arg": {
                "min_val": 0,
                "max_val": 1,
                "dtype": dtypes.float32,
                "shape": (1, 224, 224, 3),
            },
        },
        # MobileNet V2
        {
            "url": "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/3",
            "tensor_arg": {
                "min_val": 0,
                "max_val": 1,
                "dtype": dtypes.float32,
                "shape": (1, 224, 224, 3),
            },
        },
        # ResNet V1
        {
            "url": "https://tfhub.dev/google/imagenet/resnet_v1_152/classification/3",
            "tensor_arg": {
                "min_val": 0,
                "max_val": 1,
                "dtype": dtypes.float32,
                "shape": (1, 224, 224, 3),
            },
        },
        # ResNet V2
        {
            "url": "https://tfhub.dev/google/imagenet/resnet_v2_152/classification/3",
            "tensor_arg": {
                "min_val": 0,
                "max_val": 1,
                "dtype": dtypes.float32,
                "shape": (1, 224, 224, 3),
            },
        },
    ]

    for model in models:
        print("Model:", model)
        api = hub.KerasLayer(
            model["url"], trainable=False, input_shape=model["tensor_arg"]["shape"][1:]
        )  # Don't pass batch

        m = tf.keras.Sequential([api])
        m.build(model["tensor_arg"]["shape"])

        print("API: dtype:", api.dtype)
        print("API: dynamic:", api.dynamic)
        print("API: Input:", api.input)
        print("API: InputMask:", api.input_mask)
        print("API: InputShape:", api.input_shape)
        print("API: InputSpec:", api.input_spec)
        print("API: Matrics:", api.metrics)
        print("API: Name:", api.name)
        print("API: Output:", api.output)
        print("API: Output Mask:", api.output_mask)
        print("API: OutputShape:", api.output_shape)
        print("API: Sub Modules:", api.submodules)
        print("API: Config:", api.get_config())

        root = tracking.AutoTrackable()
        root.f = def_function.function(lambda x: api(x))
        concrete_func = root.f.get_concrete_function(
            tensor_spec.TensorSpec(model["tensor_arg"]["shape"], model["tensor_arg"]["dtype"])
        )

        input_data = tf.random.uniform(
            model["tensor_arg"]["shape"],
            model["tensor_arg"]["min_val"],
            model["tensor_arg"]["max_val"],
            dtype=model["tensor_arg"]["dtype"],
        )
        compare_tf_with_tvm_v2([input_data], concrete_func, atol=1e-4, rtol=1e-4)


#######################################################################
# LSTM
# ----


def _test_lstm_cell(batch_size, num_hidden, num_layers, forget_bias, dtype):
    """ One iteration of a LSTM cell """

    tf.reset_default_graph()
    input_size = num_hidden
    input_data = np.full((batch_size, input_size), 1.0, dtype=dtype)
    in_state_c = np.full((num_layers, batch_size, num_hidden), 0.1, dtype=dtype)
    in_state_h = np.full((num_layers, batch_size, num_hidden), 0.1, dtype=dtype)

    def _get_tensorflow_output():
        with tf.Session() as sess:
            with variable_scope.variable_scope(
                "root", initializer=init_ops.constant_initializer(0.5)
            ):
                m0 = array_ops.zeros([batch_size, num_hidden])
                m1 = array_ops.zeros([batch_size, num_hidden])
                x = tf.placeholder(shape=(batch_size, input_size), dtype=dtype)
                g, ((out_m0, out_m1)) = tf.contrib.rnn.LSTMBlockCell(
                    num_hidden, forget_bias=forget_bias
                )(x, ((m0, m1)))
                sess.run([variables.global_variables_initializer()])
                res = sess.run(
                    [g, out_m0, out_m1],
                    {
                        x.name: np.array([[1.0, 1.0]]),
                        m0.name: 0.1 * np.ones([batch_size, num_hidden]),
                        m1.name: 0.1 * np.ones([batch_size, num_hidden]),
                    },
                )
            graph_def = sess.graph.as_graph_def(add_shapes=True)
            final_graph_def = graph_util.convert_variables_to_constants(
                sess, graph_def, ["root/lstm_cell/LSTMBlockCell"]
            )
            return final_graph_def, res

    graph_def, tf_out = _get_tensorflow_output()
    tvm_output = run_tvm_graph(
        graph_def,
        [input_data, in_state_c, in_state_h],
        ["root/Placeholder", "root/lstm_cell/LSTMBlockCell_c", "root/lstm_cell/LSTMBlockCell_h"],
        num_output=2,
    )
    assert isinstance(tvm_output, list)

    out = tvm_output[0]
    out_state = tvm_output[1]
    out_state_tup = np.split(out_state, indices_or_sections=2, axis=1)
    out_state_c = np.reshape(out_state_tup[0], (batch_size, num_hidden))
    out_state_h = np.reshape(out_state_tup[1], (batch_size, num_hidden))
    tvm_out = [out, out_state_c, out_state_h]
    tvm.testing.assert_allclose(tf_out[0], tvm_out[0], rtol=1e-3, atol=1e-3)


#######################################################################
# PTB
# ---
# dir(tf.contrib)


def _test_forward_ptb():
    """test ptb model"""
    config = tf_testing.get_config()
    num_steps = config.num_steps
    num_hidden = config.hidden_size
    num_layers = config.num_layers
    batch_size = config.batch_size
    vocab_size = config.vocab_size
    out_sample_shape = (batch_size, vocab_size)
    out_state_shape = (num_layers, 2, batch_size, num_hidden)
    # Sample input
    inpt = "we have no useful information on"
    cnt_sample = 20

    def _pretty_print(items, is_char_model, id2word):
        if not is_char_model:
            return " ".join([id2word[x] for x in items])
        else:
            return "".join([id2word[x] for x in items]).replace("_", " ")

    def _get_tvm_graph_module(graph_def):
        # Cell inputs 'c and 'h' consist of all layers values
        shape_dict = {
            "Model/Placeholder": (batch_size, num_steps),
            "Model/RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/LSTMBlockCell_c": (
                num_layers,
                batch_size,
                num_hidden,
            ),
            "Model/RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/LSTMBlockCell_h": (
                num_layers,
                batch_size,
                num_hidden,
            ),
        }

        mod, params = relay.frontend.from_tensorflow(graph_def, shape=shape_dict)

        dtype_dict = {
            "Model/Placeholder": "int32",
            "Model/RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/LSTMBlockCell_c": "float32",
            "Model/RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/LSTMBlockCell_h": "float32",
        }
        target = "llvm"
        with relay.build_config(opt_level=0):
            graph, lib, params = relay.build(mod, target, params=params)
        from tvm.contrib import graph_runtime

        ctx = tvm.cpu(0)
        return params, graph_runtime.create(graph, lib, ctx)

    def _do_tvm_sample(model, data, in_states, params, num_samples):
        """Sampled from the model"""
        samples = []
        state = in_states
        sample = None

        def _get_sample(data, state):
            input_data = np.full((batch_size, num_steps), data, dtype="int32")
            in_state_tup = np.split(state, indices_or_sections=2, axis=1)
            in_state_c = np.reshape(in_state_tup[0], (num_layers, batch_size, num_hidden))
            in_state_h = np.reshape(in_state_tup[1], (num_layers, batch_size, num_hidden))

            model.set_input("Model/Placeholder", tvm.nd.array(input_data.astype("int32")))
            model.set_input(
                "Model/RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/LSTMBlockCell_c",
                tvm.nd.array(in_state_c.astype("float32")),
            )
            model.set_input(
                "Model/RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/LSTMBlockCell_h",
                tvm.nd.array(in_state_h.astype("float32")),
            )
            model.set_input(**params)
            model.run()
            tvm_output = model.get_output(0, tvm.nd.empty(out_sample_shape, "float32")).asnumpy()
            state_output = model.get_output(1, tvm.nd.empty(out_state_shape, "float32")).asnumpy()
            sample = tf_testing.pick_from_weight(tvm_output[0])

            return sample, state_output

        for x in data:
            sample, state = _get_sample(x, state)

        if sample is not None:
            samples.append(sample)
        else:
            samples.append(0)

        k = 1
        while k < num_samples:
            sample, state = _get_sample(samples[-1], state)
            samples.append(sample)
            k += 1
        return samples, state

    with tf.Graph().as_default():
        word_to_id, id_to_word, graph_def = tf_testing.get_workload_ptb()
        vocab_size = len(word_to_id)
        # Call the utility to import the graph definition into default graph.
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)
        sess = tf.Session()

    # TVM graph module creation
    params, m = _get_tvm_graph_module(graph_def)

    # Create 10 predicted statments of 20 words
    cnt_stm = 0
    while cnt_stm < 10:
        cnt_stm += 1
        in_state = np.full((num_layers, 2, batch_size, num_hidden), 0, dtype="float32")
        seed_for_sample = inpt.split()
        tvm_samples, tvm_state = _do_tvm_sample(
            m, [word_to_id[word] for word in seed_for_sample], in_state, params, cnt_sample
        )
        tvm_sample_str = _pretty_print(tvm_samples, False, id_to_word)
        tf_samples, tf_state = tf_testing.do_tf_sample(
            sess, [word_to_id[word] for word in seed_for_sample], in_state, cnt_sample
        )
        tf_sample_str = _pretty_print(tf_samples, False, id_to_word)
        inpt = tvm_sample_str
        tvm.testing.assert_allclose(tf_samples, tvm_samples, rtol=1e-5, atol=1e-5)
        assert tvm_sample_str == tf_sample_str


#######################################################################
# LRN (Local Response Normalization)
# ----------------------------------


def _test_lrn(ishape, size, axis, bias, alpha, beta):
    """ testing local response normalization """
    lrn_depth_radius = size / 2
    root = tracking.AutoTrackable()
    root.f = def_function.function(
        lambda x: tf.nn.local_response_normalization(
            x, depth_radius=lrn_depth_radius, bias=bias, alpha=alpha, beta=beta
        )
    )
    concrete_func = root.f.get_concrete_function(tensor_spec.TensorSpec(ishape, dtypes.float32))

    input_data = tf.random.uniform(ishape, dtype=dtypes.float32)
    compare_tf_with_tvm_v2([input_data], concrete_func)


def test_forward_lrn():
    _test_lrn((1, 3, 20, 20), 3, 1, 1.0, 1.0, 0.5)


#######################################################################
# l2_normalize
# ------------


def _test_l2_normalize(ishape, eps, axis):
    """ testing l2 normalize (uses max, sum, square, sqrt frontend operators)"""

    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x: tf.nn.l2_normalize(x, axis=axis, epsilon=eps))
    concrete_func = root.f.get_concrete_function(tensor_spec.TensorSpec(ishape, dtypes.float32))

    input_data = tf.random.uniform(ishape, dtype=dtypes.float32)
    compare_tf_with_tvm_v2([input_data], concrete_func)


def test_forward_l2_normalize():
    _test_l2_normalize((1, 3, 20, 20), 0.001, (0,))


#######################################################################
# transpose
# ---------
def _test_forward_transpose(ishape, axes=None):
    root = tracking.AutoTrackable()
    if axes is None:
        root.f = def_function.function(lambda x: tf.transpose(x))
    else:
        root.f = def_function.function(lambda x: tf.transpose(x, perm=axes))
    concrete_func = root.f.get_concrete_function(tensor_spec.TensorSpec(ishape, dtypes.float32))

    input_data = tf.random.uniform(ishape)
    compare_tf_with_tvm_v2([input_data], concrete_func)


def _test_forward_tranapose_axes_input(ishape, axes):
    root = tracking.AutoTrackable()
    axes_np = np.array(axes).astype(np.int32)
    const1 = tf.constant(axes_np, dtype=tf.int32)
    root.f = def_function.function(lambda x: tf.transpose(x, perm=tf.reverse(const1, axis=[-1])))
    concrete_func = root.f.get_concrete_function(tensor_spec.TensorSpec(ishape, dtypes.float32))

    input_data = tf.random.uniform(ishape)
    compare_tf_with_tvm_v2([input_data], concrete_func)


def test_forward_transpose():
    _test_forward_transpose((2, 3, 4), (1, 2, 0))
    _test_forward_transpose((2, 3, 4))
    _test_forward_transpose((7, 8, 8, 10))
    _test_forward_transpose((2, 3, 4), (1, 2, 0))
    _test_forward_transpose((2, 3, 4), (0, 1, 2))
    _test_forward_transpose((2, 3, 4, 5), (3, 0, 1, 2))
    _test_forward_tranapose_axes_input((2, 3, 4), (1, 2, 0))
    _test_forward_tranapose_axes_input((2, 3, 4, 5), (3, 0, 1, 2))


def _test_forward_slice_operation_input(input_value, begin_value, size_value):
    root = tracking.AutoTrackable()
    begin_tensor = tf.expand_dims(begin_value, axis=0)
    size_tensor = tf.expand_dims(size_value, axis=0)
    root.f = def_function.function(
        lambda x: tf.slice(x, begin_tensor, size_tensor, name="slice_output")
    )

    in_data = constant_op.constant(input_value, dtype="float32")

    concrete_func = root.f.get_concrete_function(
        tensor_spec.TensorSpec(np.array(input_value).shape, dtypes.float32)
    )
    compare_tf_with_tvm_v2([in_data], concrete_func)


def test_forward_slice():
    _test_forward_slice_operation_input([1, 1], 0, 2)


def test_forward_ceil():
    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x: tf.math.ceil(x))
    concrete_func = root.f.get_concrete_function(
        tensor_spec.TensorSpec([1, 3, 3, 10], dtypes.float32)
    )

    input_data = tf.random.uniform([1, 3, 3, 10])
    compare_tf_with_tvm_v2([input_data], concrete_func)


def test_forward_floor():
    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x: tf.math.floor(x))
    concrete_func = root.f.get_concrete_function(
        tensor_spec.TensorSpec([1, 3, 3, 10], dtypes.float32)
    )

    input_data = tf.random.uniform([1, 3, 3, 10])
    compare_tf_with_tvm_v2([input_data], concrete_func)


#####################################################################
# Activations
# -----------


def test_activations():
    def test_forward_activation(api, in_shape, dtype):
        root = tracking.AutoTrackable()
        root.f = def_function.function(lambda x: api(x))
        concrete_func = root.f.get_concrete_function(tensor_spec.TensorSpec([1, 3, 3, 10], dtype))

        input_data = tf.random.uniform(in_shape, -100, 100, dtype=dtype)
        compare_tf_with_tvm_v2([input_data], concrete_func)

    api_dict = {
        tf.nn.relu: [dtypes.float32, dtypes.float64, dtypes.int32],
        tf.nn.leaky_relu: [dtypes.float32, dtypes.float64, dtypes.int32],
        tf.nn.selu: [dtypes.float32, dtypes.float64],
        tf.nn.elu: [dtypes.float32, dtypes.float64],
        tf.nn.tanh: [dtypes.float32, dtypes.float64],
        tf.math.sigmoid: [dtypes.float32, dtypes.float64],
    }

    for api, api_dtypes in api_dict.items():
        for dtype in api_dtypes:
            test_forward_activation(api, (1, 3, 10, 10), dtype)


#######################################################################
# Tensor
# ------


def test_tensor_ops():
    api_dict = {
        tf.round: [
            {
                "tensor_args": [
                    {"min_val": -10, "max_val": 10, "dtype": dtypes.float32, "shape": (5, 7)}
                ],
                "ext_args": {},
            }
        ],
        tf.math.abs: [
            {
                "tensor_args": [
                    {"min_val": 1, "max_val": 100, "dtype": dtypes.float32, "shape": (9, 11)}
                ],
                "ext_args": {},
            }
        ],
        tf.math.erf: [
            {
                "tensor_args": [
                    {"min_val": -5, "max_val": 5, "dtype": dtypes.float32, "shape": (1, 3, 3, 10)}
                ],
                "ext_args": {},
            }
        ],
        tf.reverse: [
            {
                "tensor_args": [
                    {"min_val": -10, "max_val": 10, "dtype": dtypes.int32, "shape": (2, 3)}
                ],
                "ext_args": {"axis": [0]},
            },
            {
                "tensor_args": [
                    {"min_val": -10, "max_val": 10, "dtype": dtypes.float32, "shape": (2, 3, 5)}
                ],
                "ext_args": {"axis": [2]},
            },
            {
                "tensor_args": [
                    {"min_val": -10, "max_val": 10, "dtype": dtypes.float32, "shape": (2, 3, 5, 7)}
                ],
                "ext_args": {"axis": [1]},
            },
            {
                "tensor_args": [
                    {"min_val": -10, "max_val": 10, "dtype": dtypes.float64, "shape": (2, 3, 5)}
                ],
                "ext_args": {"axis": [-1]},
            },
            {
                "tensor_args": [
                    {"min_val": -10, "max_val": 10, "dtype": dtypes.float64, "shape": (2, 3, 5)}
                ],
                "ext_args": {"axis": [-3]},
            },
        ],
        tf.sign: [
            {
                "tensor_args": [
                    {"min_val": -10, "max_val": 10, "dtype": dtypes.float32, "shape": (5, 7, 11)}
                ],
                "ext_args": {},
            }
        ],
        tf.square: [
            {
                "tensor_args": [
                    {"min_val": 1, "max_val": 100, "dtype": dtypes.float32, "shape": (2, 3, 5)}
                ],
                "ext_args": {},
            }
        ],
        tf.exp: [
            {
                "tensor_args": [
                    {"min_val": -2, "max_val": 2, "dtype": dtypes.float32, "shape": (5, 7, 11)}
                ],
                "ext_args": {},
            }
        ],
        tf.pow: [
            {
                "tensor_args": [
                    {"min_val": -2, "max_val": 2, "dtype": dtypes.float32, "shape": (5, 7, 11)},
                    {"min_val": -2, "max_val": 2, "dtype": dtypes.float32, "shape": (5, 7, 11)},
                ],
                "ext_args": {},
            }
        ],
        tf.math.log: [
            {
                "tensor_args": [
                    {"min_val": 1, "max_val": 100, "dtype": dtypes.float32, "shape": (2, 3, 5)}
                ],
                "ext_args": {},
            }
        ],
        tf.math.log1p: [
            {
                "tensor_args": [
                    {"min_val": 1, "max_val": 100, "dtype": dtypes.float32, "shape": (2, 3, 5)}
                ],
                "ext_args": {},
            }
        ],
        tf.cos: [
            {
                "tensor_args": [
                    {"min_val": 1, "max_val": 100, "dtype": dtypes.float32, "shape": (2, 3, 5)}
                ],
                "ext_args": {},
            }
        ],
        tf.sin: [
            {
                "tensor_args": [
                    {"min_val": 1, "max_val": 100, "dtype": dtypes.float32, "shape": (2, 3, 5)}
                ],
                "ext_args": {},
            }
        ],
        tf.negative: [
            {
                "tensor_args": [
                    {
                        "min_val": -100,
                        "max_val": 255,
                        "dtype": dtypes.float32,
                        "shape": (224, 224, 3),
                    }
                ],
                "ext_args": {},
            }
        ],
        tf.math.log_softmax: [
            {
                "tensor_args": [
                    {"min_val": 1, "max_val": 100, "dtype": dtypes.float32, "shape": (9, 11)}
                ],
                "ext_args": {},
            }
        ],
        tf.nn.softplus: [
            {
                "tensor_args": [
                    {"min_val": 1, "max_val": 10, "dtype": dtypes.float32, "shape": (2, 3, 5)}
                ],
                "ext_args": {},
            }
        ],
        tf.math.rsqrt: [
            {
                "tensor_args": [
                    {"min_val": 1, "max_val": 100, "dtype": dtypes.float32, "shape": (5, 7, 11)}
                ],
                "ext_args": {},
            }
        ],
        tf.math.sqrt: [
            {
                "tensor_args": [
                    {"min_val": 1, "max_val": 100, "dtype": dtypes.float32, "shape": (5, 7, 11)}
                ],
                "ext_args": {},
            }
        ],
        tf.math.squared_difference: [
            {
                "tensor_args": [
                    {"min_val": -5, "max_val": 5, "dtype": dtypes.float32, "shape": (1, 3, 10, 14)},
                    {"min_val": -5, "max_val": 5, "dtype": dtypes.float32, "shape": (1, 3, 10, 14)},
                ],
                "ext_args": {},
            }
        ],
        tf.math.divide: [
            {
                "tensor_args": [
                    {"min_val": -100, "max_val": 100, "dtype": dtypes.int32, "shape": (4,)},
                    {"min_val": 1, "max_val": 100, "dtype": dtypes.int32, "shape": (4,)},
                ],
                "ext_args": {},
            },
            {
                "tensor_args": [
                    {"min_val": -100, "max_val": 100, "dtype": dtypes.float32, "shape": (4, 3, 7)},
                    {"min_val": 1, "max_val": 100, "dtype": dtypes.float32, "shape": (4, 3, 7)},
                ],
                "ext_args": {},
            },
        ],
        tf.math.floordiv: [
            {
                "tensor_args": [
                    {"min_val": -100, "max_val": 100, "dtype": dtypes.int32, "shape": (4,)},
                    {"min_val": 1, "max_val": 100, "dtype": dtypes.int32, "shape": (4,)},
                ],
                "ext_args": {},
            },
            {
                "tensor_args": [
                    {"min_val": -100, "max_val": 100, "dtype": dtypes.float32, "shape": (4, 3, 7)},
                    {"min_val": 1, "max_val": 100, "dtype": dtypes.float32, "shape": (4, 3, 7)},
                ],
                "ext_args": {},
            },
        ],
        tf.expand_dims: [
            {
                "tensor_args": [
                    {"min_val": -100, "max_val": 100, "dtype": dtypes.float32, "shape": (1,)}
                ],
                "ext_args": {"axis": 0},
            },
            {
                "tensor_args": [
                    {"min_val": -100, "max_val": 100, "dtype": dtypes.float32, "shape": (1, 2)}
                ],
                "ext_args": {"axis": 0},
            },
            {
                "tensor_args": [
                    {"min_val": -100, "max_val": 100, "dtype": dtypes.float32, "shape": (2, 3)}
                ],
                "ext_args": {"axis": -1},
            },
            {
                "tensor_args": [
                    {"min_val": -100, "max_val": 100, "dtype": dtypes.float32, "shape": (2, 3)}
                ],
                "ext_args": {"axis": 0},
            },
            {
                "tensor_args": [
                    {"min_val": -100, "max_val": 100, "dtype": dtypes.float32, "shape": (2, 3)}
                ],
                "ext_args": {"axis": 1},
            },
            {
                "tensor_args": [
                    {"min_val": -100, "max_val": 100, "dtype": dtypes.float32, "shape": (2, 3)}
                ],
                "ext_args": {"axis": -1},
            },
        ],
        tf.nn.softmax: [
            {
                "tensor_args": [
                    {"min_val": -100, "max_val": 100, "dtype": dtypes.float32, "shape": (5, 7, 11)}
                ],
                "ext_args": {"axis": 2},
            },
            {
                "tensor_args": [
                    {"min_val": -100, "max_val": 100, "dtype": dtypes.float32, "shape": (5, 7, 11)}
                ],
                "ext_args": {"axis": -1},
            },
        ],
        tf.nn.bias_add: [
            {
                "tensor_args": [
                    {
                        "min_val": -100,
                        "max_val": 100,
                        "dtype": dtypes.int32,
                        "shape": (10, 8, 16, 32),
                    },
                    {"min_val": -100, "max_val": 100, "dtype": dtypes.int32, "shape": (32,)},
                ],
                "ext_args": {},
            },
            {
                "tensor_args": [
                    {"min_val": -100, "max_val": 100, "dtype": dtypes.float32, "shape": (10, 20)},
                    {"min_val": -100, "max_val": 100, "dtype": dtypes.float32, "shape": (20,)},
                ],
                "ext_args": {},
            },
        ],
        tf.math.floormod: [
            {
                "tensor_args": [
                    {"min_val": 1, "max_val": 100, "dtype": dtypes.float32, "shape": (10,)},
                    {"min_val": 1, "max_val": 100, "dtype": dtypes.float32, "shape": (10,)},
                ],
                "ext_args": {},
            },
            {
                "tensor_args": [
                    {"min_val": -100, "max_val": 100, "dtype": dtypes.float32, "shape": (8, 2)},
                    {"min_val": 1, "max_val": 100, "dtype": dtypes.float32, "shape": (1,)},
                ],
                "ext_args": {},
            },
            {
                "tensor_args": [
                    {"min_val": 1, "max_val": 100, "dtype": dtypes.float32, "shape": (4, 3, 7)},
                    {"min_val": 1, "max_val": 100, "dtype": dtypes.float32, "shape": (4, 3, 7)},
                ],
                "ext_args": {},
            },
            {
                "tensor_args": [
                    {"min_val": -100, "max_val": 100, "dtype": dtypes.int32, "shape": (4, 3, 7)},
                    {"min_val": 1, "max_val": 100, "dtype": dtypes.int32, "shape": (4, 3, 7)},
                ],
                "ext_args": {},
            },
        ],
    }

    for api, params in api_dict.items():
        for param in params:
            _test_forward_op(api, param)


def _test_forward_right_shift(in_shape, dtype):
    """test operator RightShift"""
    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x, y: tf.bitwise.right_shift(x, y))
    concrete_func = root.f.get_concrete_function(
        tensor_spec.TensorSpec(in_shape, dtype), tensor_spec.TensorSpec(in_shape, dtype)
    )

    lh_data = tf.random.uniform(in_shape, 1, 3, dtype=dtype)
    rh_data = tf.random.uniform(in_shape, 1, 8, dtype=dtype)
    compare_tf_with_tvm_v2([lh_data, rh_data], concrete_func)


def test_forward_right_shift():
    _test_forward_right_shift((7,), dtypes.int32)
    _test_forward_right_shift((3, 11), dtypes.int64)


def _test_forward_left_shift(in_shape, dtype):
    """test operator LeftShift"""
    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x, y: tf.bitwise.left_shift(x, y))
    concrete_func = root.f.get_concrete_function(
        tensor_spec.TensorSpec(in_shape, dtype), tensor_spec.TensorSpec(in_shape, dtype)
    )

    lh_data = tf.random.uniform(in_shape, 1, 3, dtype=dtype)
    rh_data = tf.random.uniform(in_shape, 1, 8, dtype=dtype)
    compare_tf_with_tvm_v2([lh_data, rh_data], concrete_func)


def test_forward_left_shift():
    _test_forward_left_shift((10,), dtypes.int32)
    _test_forward_left_shift((224, 224, 3), dtypes.int64)


#######################################################################
# Size
# ----


def test_forward_size():
    def check_size(ishape):
        tf_input_shape = list(ishape)
        tf_input_shape[0] = None

        root = tracking.AutoTrackable()
        root.f = def_function.function(lambda x: tf.size(x))
        concrete_func = root.f.get_concrete_function(
            tensor_spec.TensorSpec(tf_input_shape, dtypes.float32)
        )

        input_data = tf.random.uniform(ishape, dtype=dtypes.float32)
        compare_tf_with_tvm_v2([input_data], concrete_func)

    check_size((10, 8, 16, 32))
    check_size((10,))


#######################################################################
# Relational operators
# --------------------


def test_forward_rel_ops():
    params = [
        {
            "tensor_args": [
                {"min_val": -10, "max_val": 10, "dtype": dtypes.float32, "shape": (5, 7, 9)},
                {"min_val": -10, "max_val": 10, "dtype": dtypes.float32, "shape": (5, 7, 9)},
            ],
            "ext_args": {},
        },
        {
            "tensor_args": [
                {"min_val": -10, "max_val": 10, "dtype": dtypes.int32, "shape": (5, 7, 9)},
                {"min_val": -10, "max_val": 10, "dtype": dtypes.int32, "shape": (5, 7, 9)},
            ],
            "ext_args": {},
        },
    ]

    for api in [
        tf.math.less,
        tf.math.greater,
        tf.math.less_equal,
        tf.math.greater_equal,
        tf.math.equal,
        tf.math.not_equal,
    ]:
        for param in params:
            _test_forward_op(api, param)


#######################################################################
# Maximum, Minimum
# ----------------
def test_forward_maximum():
    """test Op Maximum"""

    def check_maximum(lh_shape, rh_shape, dtype):
        root = tracking.AutoTrackable()
        root.f = def_function.function(lambda x, y: tf.math.maximum(x, y))
        concrete_func = root.f.get_concrete_function(
            tensor_spec.TensorSpec(lh_shape, dtype), tensor_spec.TensorSpec(rh_shape, dtype)
        )

        lh_data = tf.random.uniform(lh_shape, -100, 100, dtype=dtype)
        rh_data = tf.random.uniform(rh_shape, -100, 100, dtype=dtype)
        compare_tf_with_tvm_v2([lh_data, rh_data], concrete_func)

    check_maximum((10, 8, 16, 32), (1,), dtype=dtypes.int32)
    check_maximum((10, 8, 16, 32), (10, 8, 16, 32), dtype=dtypes.float32)


def test_forward_minimum():
    """test Op Minimum"""

    def check_minimum(lh_shape, rh_shape, dtype):
        root = tracking.AutoTrackable()
        root.f = def_function.function(lambda x, y: tf.math.minimum(x, y))
        concrete_func = root.f.get_concrete_function(
            tensor_spec.TensorSpec(lh_shape, dtype), tensor_spec.TensorSpec(rh_shape, dtype)
        )

        lh_data = tf.random.uniform(lh_shape, -100, 100, dtype=dtype)
        rh_data = tf.random.uniform(rh_shape, -100, 100, dtype=dtype)
        compare_tf_with_tvm_v2([lh_data, rh_data], concrete_func)

    check_minimum((10, 8, 16, 32), (1,), dtype=dtypes.int32)
    check_minimum((10, 8, 16, 32), (10, 8, 16, 32), dtype=dtypes.float32)


#######################################################################
# Multi Input to graph
# --------------------


def test_forward_multi_input():
    root = tracking.AutoTrackable()

    root.f = def_function.function(
        lambda a, b, c, d: tf.multiply(tf.add(a, b), tf.math.subtract(c, d))
    )
    concrete_func = root.f.get_concrete_function(
        tensor_spec.TensorSpec((3, 3), dtypes.int32),
        tensor_spec.TensorSpec((3, 3), dtypes.int32),
        tensor_spec.TensorSpec((3, 3), dtypes.int32),
        tensor_spec.TensorSpec((3, 3), dtypes.int32),
    )

    indata1 = tf.random.uniform((3, 3), -100, 100, dtype=dtypes.int32)
    indata2 = tf.random.uniform((3, 3), -100, 100, dtype=dtypes.int32)
    indata3 = tf.random.uniform((3, 3), -100, 100, dtype=dtypes.int32)
    indata4 = tf.random.uniform((3, 3), -100, 100, dtype=dtypes.int32)
    compare_tf_with_tvm_v2([indata1, indata2, indata3, indata4], concrete_func)


#######################################################################
# Multi Output to Graph
# ---------------------


def test_forward_multi_output():
    root = tracking.AutoTrackable()

    @def_function.function
    def func(a, b, c, d):
        out1 = tf.add(a, b)
        out2 = tf.math.subtract(c, d)
        return out1, out2

    root.f = def_function.function(lambda a, b, c, d: (tf.add(a, b), tf.math.subtract(c, d)))
    concrete_func = root.f.get_concrete_function(
        tensor_spec.TensorSpec((3, 3), dtypes.int32),
        tensor_spec.TensorSpec((3, 3), dtypes.int32),
        tensor_spec.TensorSpec((3, 3), dtypes.int32),
        tensor_spec.TensorSpec((3, 3), dtypes.int32),
    )

    indata1 = tf.random.uniform((3, 3), -100, 100, dtype=dtypes.int32)
    indata2 = tf.random.uniform((3, 3), -100, 100, dtype=dtypes.int32)
    indata3 = tf.random.uniform((3, 3), -100, 100, dtype=dtypes.int32)
    indata4 = tf.random.uniform((3, 3), -100, 100, dtype=dtypes.int32)
    compare_tf_with_tvm_v2([indata1, indata2, indata3, indata4], concrete_func)


#######################################################################
# Variable
# --------


def _test_variable(data):
    """ One iteration of a variable """
    size = data.shape[1]

    with variable_scope.variable_scope("linear", reuse=None):
        w = variable_scope.get_variable("w", shape=[size, size], dtype=data.dtype)
    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x: tf.matmul(x, w))
    concrete_func = root.f.get_concrete_function(
        tensor_spec.TensorSpec(data.shape, dtype=data.dtype)
    )

    input_data = tf.convert_to_tensor(data)
    compare_tf_with_tvm_v2([input_data], concrete_func)


def test_forward_variable():
    """Variable type op test"""
    _test_variable(np.random.uniform(size=(32, 100)).astype("float32"))


#######################################################################
# OneHot
# ----------------------


def _test_forward_one_hot(indices_shape, depth, on_value, off_value, axis, out_dtype):
    root = tracking.AutoTrackable()
    root.f = def_function.function(
        lambda x: tf.one_hot(x, depth, on_value, off_value, axis, dtype=out_dtype)
    )

    concrete_func = root.f.get_concrete_function(
        tensor_spec.TensorSpec(indices_shape, dtypes.int32)
    )
    input_data = tf.random.uniform(indices_shape, 0, 5, dtype=dtypes.int32)
    compare_tf_with_tvm_v2([input_data], concrete_func)


def test_forward_one_hot():
    _test_forward_one_hot((3,), 3, 1, 0, -1, dtypes.int32)
    _test_forward_one_hot((3,), 3, 1.0, 0.0, -1, dtypes.float32)
    _test_forward_one_hot((2, 2), 5, 2, -2, 0, dtypes.int32)
    _test_forward_one_hot((2, 2), 5, 0.5, -0.5, 1, dtypes.float32)
    _test_forward_one_hot((3, 2, 4, 5), 6, 1, 0, 1, dtypes.int32)
    _test_forward_one_hot((3, 2, 4, 5), 6, 1.0, 0.0, 0, dtypes.float32)


#######################################################################
# AddN
# ----


'''
def _test_forward_add_n(inputs):
    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x: tf.add_n(x))

    concrete_func = root.f.get_concrete_function(tensor_spec.TensorSpec((3, 3, 3), dtypes.int32))
    input_data = tf.random.uniform((3, 3, 3), 0, 5, dtype=dtypes.int32)
    compare_tf_with_tvm_v2([input_data], concrete_func)


def test_forward_add_n():
    x = np.random.randint(1, 100, size=(3, 3, 3), dtype=np.int32)
    y = np.random.randint(1, 100, size=(3, 3, 3), dtype=np.int32)
    z = np.random.randint(1, 100, size=(3, 3, 3), dtype=np.int32)
    m, n, o = x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)
    in0 = x
    in1 = [x, y]
    in2 = (x, y, z)
    in3 = m
    in4 = [m, n]
    in5 = (m, n, o)
    _test_forward_add_n(in0)
    # _test_forward_add_n(in1)
    # _test_forward_add_n(in2)
    # _test_forward_add_n(in3)
    # _test_forward_add_n(in4)
    # _test_forward_add_n(in5)
'''


#######################################################################
# Main
# ----
if __name__ == "__main__":
    # GPU Global Configuration
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        # Restrict TensorFlow to only allocate 2GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
            )
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
            pass

    # Transforms
    test_forward_slice()
    test_forward_transpose()
    test_forward_reshape()
    test_forward_depthtospace()
    test_forward_spacetodepth()
    test_forward_squeeze()
    test_forward_pack()
    test_forward_size()
    test_forward_broadcast_to()
    test_forward_fill()
    test_forward_crop()
    # test_forward_resize() -  TODO
    test_forward_crop_and_resize()
    test_forward_pad()
    test_forward_unpack()
    test_forward_gather()
    test_forward_gather_nd()
    # test_forward_stridedslice() - TODO
    test_forward_split()
    test_forward_tile()
    test_forward_top_k_v2()
    test_forward_maximum()
    test_forward_minimum()
    test_forward_range()
    test_forward_truncatemod()
    test_forward_one_hot()

    # Activations
    test_activations()

    # Tensor
    test_forward_ceil()
    test_forward_floor()
    test_forward_right_shift()
    test_forward_left_shift()
    # test_forward_add_n() - TODO
    test_tensor_ops()

    # Reductions
    test_forward_reduce_all()
    test_forward_reduce_any()
    test_reductions()

    # Relational ops
    test_forward_rel_ops()
    test_forward_logical()
    test_forward_where()
    test_forward_matmul()
    test_forward_batch_matmul()

    # Array Ops
    # TODO:  TensorListGetItem, TensorListSetItem, TensorListReserve,
    #       TensorListScatterIntoExistingList, TensorListSplit,
    #       TensorListConcatV2, TensorListLength, TensorListFromTensor
    # test_tensor_array_constructor()
    # test_tensor_array_scatter()
    # test_tensor_array_split()
    # test_tensor_array_concat()
    # test_tensor_array_size()
    # test_tensor_array_unstack()

    # General
    test_forward_multi_input()
    test_forward_multi_output()
    test_forward_variable()

    # NN
    test_forward_convolution()
    test_forward_convolution3d()
    test_forward_pooling()
    test_forward_lrn()
    test_forward_l2_normalize()
    test_forward_biasadd()
    test_forward_space_to_batch_nd()
    # test_forward_batch_to_space_nd() - TODO

    # RNN
    # test_forward_lstm() - TODO

    # End to End
    # test_forward_image_classification_models() # TODO: Need docker update.
    # test_forward_ptb() - TODO
