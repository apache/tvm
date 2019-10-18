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
The Tensorflow frontend helps in importing TensorFlow models into TVM.

Supported versions:
- 1.12 and below

Tested models:
- Inception (V1/V2/V3/V4)
- Resnet (All)
- Mobilenet (V1/V2 All)
- Vgg (16/19)
- BERT (Base/3-layer)

## Preparing a Model for Inference

### Remove Unneeded Nodes

The export process will remove many nodes that are not needed for inference, but unfortunately will leave some remaining.
The nodes that should be manually removed are:
- Dropout, including the [dropout node](https://www.tensorflow.org/api_docs/python/tf/nn/dropout) and [dropout wrapper](https://www.tensorflow.org/versions/r1.12/api_docs/python/tf/nn/rnn_cell/DropoutWrapper?hl=hr)
- Assert

### Convert None Dimensions to Constants

TVM has minimal support for dynamic tensor shapes. Dimensions that are ```None``` should be replaced with constants. For example, a model may accept an input with shape ```(None,20)```. This should be converted to something like ```(1,20)```. The model should be modified accordingly to ensure that these shapes match throughout the graph.

### Export

TensorFlow frontend expects a freezed protobuf (.pb) or saved model as input. It currently does not support checkpoint (.ckpt).
The graphdef needed to pass into the TensorFlow frontend can be extracted using the [TFParser](https://github.com/dmlc/tvm/blob/77445311540c0dfa7b124304b5cf89da6f2c210f/python/tvm/relay/frontend/tensorflow_parser.py) helper class.

The model should be exported with a number of transformations to prepare the model for inference. It is also important to set ```add_shapes=True```. This will embed the output shapes of each node into the graph. Here is one function to export a model as a protobuf given a session:

```TODO: code example```

Another method is to [export and freeze the graph](https://github.com/tensorflow/models/tree/master/research/slim#exporting-the-inference-graph).

## Import the Model

## Explicit Shape:
```add_shapes=True``` might not provide sufficient shape information. Passing an explicit dictionary of input names to shape in ```from_tensorflow``` will help ensure that the shapes can be known throughout the entire graph. Please refer to these [test cases](https://github.com/dmlc/tvm/blob/master/nnvm/tests/python/frontend/tensorflow/test_forward.py#L36) as an example.

## Data Layout
Most TensorFlow models are released with NHWC layout. NCHW layout often provides better performance, especially on GPU. The TensorFlow frontend can automatically convert the model's data layout by passing the argument ```layout='NCHW'``` to ```from_tensorflow```.

## Best Practices

- Use static tensor shapes instead of dynamic shapes (remove ```None``` dimensions).
- Use static RNN instead of dynamic RNN, as ```TensorArray``` isn't supported yet.

## Supported Ops


