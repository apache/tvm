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
"""
BatchNorm without given mean and variance given testcases
====================
This is a test script to test fused_batch_norm operators
in TensorFlow frontend when mean and variance are not given.
"""
import tvm
import numpy as np
import tensorflow as tf
from tvm import relay
from tensorflow.python.framework import graph_util

def test_sharing_node():
    g = tf.Graph()
    with g.as_default():
        input_tensor = tf.placeholder(tf.float32, shape=(2, 2, 2), name='input')
        axis = tf.constant([-1], dtype=tf.int32, name='axis')
        mean0 = tf.reduce_mean(input_tensor, axis=axis, keepdims=False, name='mean0')
        mean1 = tf.reduce_mean(input_tensor, axis=axis, keepdims=False, name='mean1')
        sum = tf.add(mean0, mean1, name='sum')
        out = tf.identity(sum, name='output')
        LOGDIR='./before_dir'
        train_writer = tf.summary.FileWriter(LOGDIR)
        train_writer.add_graph(out.graph)
    data = np.random.rand(2, 2, 2)
    with tf.Session(graph=out.graph) as sess:
        sess.run([tf.global_variables_initializer()])
        tf_out = sess.run(out, feed_dict={input_tensor:data})
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['output'])

    for device in ["llvm"]:
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            continue
        mod, params = relay.frontend.from_tensorflow(constant_graph,
                                                     outputs=['output'])
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(mod,
                                             target=device,
                                             params=params)
        from tvm.contrib import graph_runtime
        m = graph_runtime.create(graph, lib, ctx)
        m.set_input(**params)
        m.set_input('input', data)
        m.run()
        tvm_out = m.get_output(0)
        tvm.testing.assert_allclose(tvm_out.asnumpy(), tf_out.astype(tvm_out.dtype),
                                    atol=1e-3, rtol=1e-3)

if __name__ == "__main__":
    test_sharing_node()
