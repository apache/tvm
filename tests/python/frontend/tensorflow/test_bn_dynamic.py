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
try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf
from tvm import relay
from tensorflow.python.framework import graph_util

def verify_fused_batch_norm(shape):
    g = tf.Graph()
    with g.as_default():
        input_tensor = tf.placeholder(tf.float32, shape=shape, name='input')
        alpha = tf.constant(np.random.rand(shape[-1],), dtype=tf.float32, name='alpha')
        beta = tf.constant(np.random.rand(shape[-1],), dtype=tf.float32, name='beta')
        bn = tf.nn.fused_batch_norm(x=input_tensor, offset=beta, scale=alpha, name='bn')
        out = tf.identity(bn[0], name='output')
    data = np.random.rand(*shape)
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

def test_fused_batch_norm():
    verify_fused_batch_norm(shape=(1, 12, 12, 32))
    verify_fused_batch_norm(shape=(1, 24, 24, 64))
    verify_fused_batch_norm(shape=(1, 64, 64, 128))
    verify_fused_batch_norm(shape=(8, 12, 12, 32))
    verify_fused_batch_norm(shape=(16, 12, 12, 32))
    verify_fused_batch_norm(shape=(32, 12, 12, 32))

if __name__ == "__main__":
    test_fused_batch_norm()
