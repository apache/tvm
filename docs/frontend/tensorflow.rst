..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

TensorFlow Frontend
===================

The TensorFlow frontend helps in importing TensorFlow models into TVM.

Supported versions:

- 1.12 and below

Tested models:

- Inception (V1/V2/V3/V4)
- Resnet (All)
- Mobilenet (V1/V2 All)
- Vgg (16/19)
- BERT (Base/3-layer)

Preparing a Model for Inference
-------------------------------

Remove Unneeded Nodes
~~~~~~~~~~~~~~~~~~~~~

The export process will remove many nodes that are not needed for inference, but unfortunately will leave some remaining. The nodes that should be manually removed are:

- Dropout, including `Dropout`_ and `DropoutWrapper`_
- `Assert`_

.. _Dropout: https://www.tensorflow.org/api_docs/python/tf/nn/dropout
.. _DropoutWrapper: https://www.tensorflow.org/versions/r1.12/api_docs/python/tf/nn/rnn_cell/DropoutWrapper?hl=hr
.. _Assert: https://www.tensorflow.org/api_docs/python/tf/debugging/Assert

Convert None Dimensions to Constants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TVM has minimal support for dynamic tensor shapes. Dimensions that are ``None`` should be replaced with constants. For example, a model may accept an input with shape ``(None,20)``. This should be converted to a shape like ``(1,20)``. The model should be modified accordingly to ensure that these shapes match throughout the graph.

Export
~~~~~~

TensorFlow frontend expects a frozen protobuf (.pb) or saved model as input. It currently does not support checkpoint (.ckpt). The graphdef needed by the TensorFlow frontend can be extracted from the active session, or by using the `TFParser`_ helper class.

.. _TFParser: https://github.com/apache/incubator-tvm/blob/master/python/tvm/relay/frontend/tensorflow_parser.py

The model should be exported with a number of transformations to prepare the model for inference. It is also important to set ```add_shapes=True```, as this will embed the output shapes of each node into the graph. Here is one function to export a model as a protobuf given a session:

.. code:: python

    import tensorflow as tf
    from tensorflow.tools.graph_transforms import TransformGraph

    def export_pb(session):
        with tf.gfile.GFile("myexportedmodel.pb", "wb") as f:
            inputs = ["myinput1", "myinput2"] # replace with your input names
            outputs = ["myoutput1"] # replace with your output names
            graph_def = session.graph.as_graph_def(add_shapes=True)
            graph_def = tf.graph.util.convert_variables_to_constants(session, graph_def, outputs)
            graph_def = TransformGraph(
                graph_def,
                inputs,
                outputs,
                [
                    "remove_nodes(op=Identity, op=CheckNumerics, op=StopGradient)",
                    "sort_by_execution_order", # sort by execution order after each transform to ensure correct node ordering
                    "remove_device",
                    "sort_by_execution_order",
                    "fold_batch_norms",
                    "sort_by_execution_order",
                    "fold_old_batch_norms",
                    "sort_by_execution_order"
                ]
            )
            f.write(graph_def.SerializeToString())

Another method is to `export and freeze the graph <https://github.com/tensorflow/models/tree/master/research/slim#exporting-the-inference-graph>`_.

Import the Model
----------------

Explicit Shape:
~~~~~~~~~~~~~~~

To ensure shapes can be known throughout the entire graph, pass the ```shape``` argument to ```from_tensorflow```. This dictionary maps input names to input shapes. Please refer to these `test cases <https://github.com/apache/incubator-tvm/blob/master/tests/python/frontend/tensorflow/test_forward.py#L36>`_ for examples.

Data Layout
~~~~~~~~~~~

Most TensorFlow models are released with NHWC layout. NCHW layout often provides better performance, especially on GPU. The TensorFlow frontend can automatically convert the model's data layout by passing the argument ```layout='NCHW'``` to ```from_tensorflow```.

Best Practices
--------------

- Use static tensor shapes instead of dynamic shapes (remove ```None``` dimensions).
- Use static RNN instead of dynamic RNN, as ```TensorArray``` isn't supported yet.

Supported Ops
-------------

- Abs
- Add
- AddN
- All
- Any
- ArgMax
- ArgMin
- AvgPool
- BatchMatMul
- BatchMatMulV2
- BatchNormWithGlobalNormalization
- BatchToSpaceND
- BiasAdd
- BroadcastTo
- Cast
- Ceil
- CheckNumerics
- ClipByValue
- Concat
- ConcatV2
- Conv2D
- Cos
- Tan
- CropAndResize
- DecodeJpeg
- DepthwiseConv2dNative
- DepthToSpace
- Dilation2D
- Equal
- Elu
- Enter
- Erf
- Exit
- Exp
- ExpandDims
- Fill
- Floor
- FloorDiv
- FloorMod
- FusedBatchNorm
- FusedBatchNormV2
- Gather
- GatherNd
- GatherV2
- Greater
- GreaterEqual
- Identity
- IsFinite
- IsInf
- IsNan
- LeakyRelu
- LeftShift
- Less
- LessEqual
- Log
- Log1p
- LoopCond
- LogicalAnd
- LogicalOr
- LogicalNot
- LogSoftmax
- LRN
- LSTMBlockCell
- MatMul
- Max
- MaxPool
- Maximum
- Mean
- Merge
- Min
- Minimum
- MirrorPad
- Mod
- Mul
- Neg
- NextIteration
- NotEqual
- OneHot
- Pack
- Pad
- PadV2
- Pow
- Prod
- Range
- Rank
- RealDiv
- Relu
- Relu6
- Reshape
- ResizeBilinear
- ResizeBicubic
- ResizeNearestNeighbor
- ReverseV2
- RightShift
- Round
- Rsqrt
- Select
- Selu
- Shape
- Sigmoid
- Sign
- Sin
- Size
- Slice
- Softmax
- Softplus
- SpaceToBatchND
- SpaceToDepth,
- Split
- SplitV
- Sqrt
- Square
- SquareDifference
- Squeeze
- StridedSlice
- Sub
- Sum
- Switch
- Tanh
- TensorArrayV3
- TensorArrayScatterV3
- TensorArrayGatherV3
- TensorArraySizeV3
- TensorArrayWriteV3
- TensorArrayReadV3
- TensorArraySplitV3
- TensorArrayConcatV3
- Tile
- TopKV2
- Transpose
- TruncateMod
- Unpack
- UnravelIndex
- Where
- ZerosLike
