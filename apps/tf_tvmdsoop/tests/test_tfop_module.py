#!/usr/bin/env python

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
"""Test script for tf op module"""
import tempfile
import os
import logging
import tensorflow as tf
import numpy as np
import tvm
from tvm import te
from tvm.contrib import tf_op


def test_use_tvmdso_op():
    """main test function"""

    def export_cpu_add_lib():
        """create cpu add op lib"""
        n = te.var("n")
        ph_a = te.placeholder((n,), name="ph_a")
        ph_b = te.placeholder((n,), name="ph_b")
        ph_c = te.compute(ph_a.shape, lambda i: ph_a[i] + ph_b[i], name="ph_c")
        sched = te.create_schedule(ph_c.op)
        fadd_dylib = tvm.build(sched, [ph_a, ph_b, ph_c], "c", name="vector_add")
        lib_path = tempfile.mktemp("tvm_add_dll.so")
        fadd_dylib.export_library(lib_path)
        return lib_path

    def export_gpu_add_lib():
        """create gpu add op lib"""
        n = te.var("n")
        ph_a = te.placeholder((n,), name="ph_a")
        ph_b = te.placeholder((n,), name="ph_b")
        ph_c = te.compute(ph_a.shape, lambda i: ph_a[i] + ph_b[i], name="ph_c")
        sched = te.create_schedule(ph_c.op)
        b_axis, t_axis = sched[ph_c].split(ph_c.op.axis[0], factor=64)
        sched[ph_c].bind(b_axis, te.thread_axis("blockIdx.x"))
        sched[ph_c].bind(t_axis, te.thread_axis("threadIdx.x"))
        fadd_dylib = tvm.build(sched, [ph_a, ph_b, ph_c], "cuda", name="vector_add")
        lib_path = tempfile.mktemp("tvm_add_cuda_dll.so")
        fadd_dylib.export_library(lib_path)
        return lib_path

    def test_add(session, lib_path, tf_device):
        """test add lib with TensorFlow wrapper"""
        module = tf_op.OpModule(lib_path)

        left = tf.placeholder("float32", shape=[4])
        right = tf.placeholder("float32", shape=[4])

        feed_dict = {left: [1.0, 2.0, 3.0, 4.0], right: [5.0, 6.0, 7.0, 8.0]}
        expect = np.asarray([6.0, 8.0, 10.0, 12.0])

        add1 = module.func("vector_add", output_shape=[4], output_dtype="float")
        add2 = module.func("vector_add", output_shape=tf.shape(left), output_dtype="float")
        add3 = module.func("vector_add", output_shape=[tf.shape(left)[0]], output_dtype="float")

        with tf.device(tf_device):
            output1 = session.run(add1(left, right), feed_dict)
            np.testing.assert_equal(output1, expect)

            output2 = session.run(add2(left, right), feed_dict)
            np.testing.assert_equal(output2, expect)

            output3 = session.run(add3(left, right), feed_dict)
            np.testing.assert_equal(output3, expect)

    def cpu_test(session):
        """test function for cpu"""
        cpu_lib = None
        try:
            cpu_lib = export_cpu_add_lib()
            test_add(session, cpu_lib, "/cpu:0")
        finally:
            if cpu_lib is not None:
                os.remove(cpu_lib)

    def gpu_test(session):
        """test function for gpu"""
        gpu_lib = None
        try:
            gpu_lib = export_gpu_add_lib()
            test_add(session, gpu_lib, "/gpu:0")
        finally:
            if gpu_lib is not None:
                os.remove(gpu_lib)

    with tf.Session() as session:
        if tvm.runtime.enabled("cpu"):
            logging.info("Test TensorFlow op on cpu kernel")
            cpu_test(session)
        if tvm.runtime.enabled("gpu"):
            logging.info("Test TensorFlow op on gpu kernel")
            gpu_test(session)


if __name__ == "__main__":
    test_use_tvmdso_op()
