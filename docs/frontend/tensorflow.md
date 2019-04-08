<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Tensorflow Frontend
Tensorflow frontend helps in importing tensorflow released model into TVM.

This document helps few steps while importing various different models from
[tensorflow research/slim](https://github.com/tensorflow/models/tree/master/research/slim).

Current frontend is tested with all versions of below models
- Inception (V1/V2/V3/V4)
- Resnet (All)
- Mobilenet (V1/V2 All)
- Vgg (16/19)

Tensorflow frontend expects a freezed protobuf format as input.

Not all models are released as freezed protobuf. Some of them are checkpoints (.ckpt).
Please refer to [export](https://github.com/tensorflow/models/tree/master/research/slim#exporting-the-inference-graph) 
and [freeze](https://github.com/tensorflow/models/tree/master/research/slim#freezing-the-exported-graph) 
instructions to generate protobuf from checkpoint.

## General Instructions

### Add Shapes:
While freezing of protobuf add additional option ```add_shapes=True``` to embed output shapes of each node into graph.
You may use ```tvm.relay.testing.tf.AddShapesToGraphDef``` from nnvm for the same.
Please refer to [tensorflow tutorial](https://github.com/dmlc/tvm/blob/master/tutorials/nnvm/from_tensorflow.py).

### Explicit Shape:
There might be situations where the add_shapes=True may not provide sufficient information about shape.
You may pass explicit dictionary of input shapes argument for ```from_tensorflow```.
Please refer to [test cases](https://github.com/dmlc/tvm/blob/master/nnvm/tests/python/frontend/tensorflow/test_forward.py#L36).

### GPU:
Most of these tensorflow models are released for CPU with NHWC layout.
To compile for GPU we need to pass extra argument ```layout='NCHW'``` for from_tensorflow.
This option will do a layout conversion before and after for neural network ops.
Remaining nnvm build options for GPU compilation remain as it is.
