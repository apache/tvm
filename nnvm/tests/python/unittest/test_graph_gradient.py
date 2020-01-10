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
import nnvm.symbol as sym
from nnvm.compiler import graph_util

def test_cnn_gradients():
    # input data
    h = 128
    w = 128
    data_shape = (1000, 3, h, w)
    data = sym.Variable('data', shape=data_shape, dtype=0)

    # conv2d
    num_channels = 64
    kernel_size = 32
    conv_w_shape = (num_channels, 3, kernel_size, kernel_size)
    conv_b_shape = (num_channels,)
    conv_w = sym.Variable('conv_w', shape=conv_w_shape)
    conv_b = sym.Variable('conv_b', shape=conv_b_shape)
    conv1 = sym.conv2d(data=data, weight=conv_w, bias=conv_b,
                      channels=num_channels, kernel_size=(kernel_size, kernel_size),
                      name='conv1')
    # relu1
    relu1 = sym.relu(data=conv1, name='relu1')
    # max pooling
    max_pooling1 = sym.max_pool2d(data=relu1, pool_size=(2, 2), name='max_pooling1')
    # flatten
    flatten1 = sym.flatten(data=max_pooling1)
    # shape after flatten
    flatten_out_shape = (h - kernel_size) * (w - kernel_size) * num_channels
    # dense1
    dense1_hidden_units = 100
    dense1 = sym.dense(data=flatten1, name='dense1', units=dense1_hidden_units)
    # relu2
    relu2 = sym.relu(data=dense1, name='relu2')
    # dense2
    dense2_hidden_units = 10
    dense2 = sym.dense(data=relu2, name='dense2', units=dense2_hidden_units)
    # softmax
    mlp = sym.softmax(data=dense2, name='softmax')
    # fake non-sparse label
    label = sym.full_like(mlp, fill_value=1)
    # cross entropy loss
    ce_loss = sym.sum(
        sym.elemwise_mul(sym.log_softmax(dense2), label),
        axis=1,
        keepdims=True,
        name="ce_loss")

    # input variables:
    # print grad_g.symbol.list_input_names()
    # >> ['data', 'conv_w', 'conv_b',
    #     'dense1_weight', 'dense1_bias',
    #     'dense2_weight', 'dense2_bias']

    # output gradient variables:
    # print grad_g.symbol.list_output_names()
    # >> ['conv1_grad_data', 'conv1_grad_weight', 'conv1_grad_bias',
    #     'dense1_grad_weight', 'dense1_grad_bias',
    #     'dense2_grad_weight', 'dense2_grad_bias']
    grad_g = graph_util.get_gradient_graph(ce_loss, ce_loss.list_input_variables())

    # infer shape
    in_shapes, out_shapes = graph_util.infer_shape(grad_g)

    # forward graph shape
    assert in_shapes == [list(data_shape), list(conv_w_shape), list(conv_b_shape),
                          [dense1_hidden_units, flatten_out_shape], [dense1_hidden_units],
                          [dense2_hidden_units, dense1_hidden_units], [dense2_hidden_units]]
    # input grads shape should be equal with input shape
    assert in_shapes == out_shapes

    # output grads w.r.t input variables
    grads = graph_util.gradients(ce_loss, ce_loss.list_input_variables())

    # gradients number should be equal with grad_input number
    assert len(grads) == len(ce_loss.list_input_variables())

    # infer type
    in_dtypes, out_dtypes = graph_util.infer_dtype(grad_g)
    assert out_dtypes == ['float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32']

def test_multi_loss_graph_gradients():
    # input data
    shape1 = (1000, 100)
    data1 = sym.Variable('data1', shape=(1000, 100), dtype=0)

    # fake non-sparse label
    label = sym.full(fill_value=3)

    # square loss
    sub1 = sym.elemwise_sub(data1, label, name="sub1")
    square_loss = sym.sum(data=sub1**2, axis=1, name="square_loss")

    # fake loss1
    shape2 = (1000, )
    data2 = sym.Variable('data2', shape=shape2, dtype=0)
    loss1 = sym.sqrt(data2, name="loss1")

    # fake loss2
    loss2 = sym.relu(data1, name='loss2')

    # block loss1
    total_loss = sym.elemwise_sum(
        sym.block_grad(loss1),
        square_loss,
        num_args=2,
        name="total_loss")

    # grad_g.symbol.list_output_names()
    # >> ['loss1_grad_0_output', 'grad_sum_output']
    grad_g = graph_util.get_gradient_graph([total_loss, loss2], total_loss.list_input_variables())
    # infer shape
    in_shapes, out_shapes = graph_util.infer_shape(grad_g)
    assert out_shapes == [list(shape2), list(shape1)]

    # grad_data1 is elemwise_sum of grad_loss2, grad_square_loss
    grad_data1 = grad_g.symbol[1]
    assert grad_data1.list_attr()['num_args'] == '2'

    # block grad should return zero grad
    grad_data2 = grad_g.symbol[0]
    assert 'zeros_like' in grad_g.ir()

    # test reverse infer shape for label
    assert grad_g.apply('InferShape').json_attr('shape_num_unknown_nodes') == 0

    # infer type
    in_dtypes, out_dtypes = graph_util.infer_dtype(grad_g)
    assert out_dtypes == ['float32', 'float32']

    # test reverse infer type for label
    assert grad_g.apply('InferType').json_attr('dtype_num_unknown_nodes') == 0


if __name__ == "__main__":
    test_cnn_gradients()
    test_multi_loss_graph_gradients()
