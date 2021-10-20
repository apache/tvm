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
# pylint: disable-msg=too-many-arguments, too-many-locals, assignment-from-no-return
""" Conv Int8 functional and performance testing"""
import sys
import logging
import numpy as np
import tvm
from tvm import te
from tvm import topi

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
LOGGER = logging.getLogger("test_conv_int8_intel")
LOGGER.disabled = False

# All the WORKLOADS from Resnet except first layer
# Workload is ['height', 'width', 'in_filter', 'out_filter',
#              'hkernel', 'wkernel', 'hpad', 'wpad', 'hstride', 'wstride'])
WORKLOADS = [
    (56, 56, 64, 64, 3, 3, 1, 1, 1, 1),
    (56, 56, 64, 64, 1, 1, 0, 0, 1, 1),
    (56, 56, 64, 128, 3, 3, 1, 1, 2, 2),
    (56, 56, 64, 128, 1, 1, 0, 0, 2, 2),
    (28, 28, 128, 128, 3, 3, 1, 1, 1, 1),
    (28, 28, 128, 256, 3, 3, 1, 1, 2, 2),
    (28, 28, 128, 256, 1, 1, 0, 0, 2, 2),
    (14, 14, 256, 256, 3, 3, 1, 1, 1, 1),
    (14, 14, 256, 512, 3, 3, 1, 1, 2, 2),
    (14, 14, 256, 512, 1, 1, 0, 0, 2, 2),
    (7, 7, 512, 512, 3, 3, 1, 1, 1, 1),
    (56, 56, 64, 256, 1, 1, 0, 0, 1, 1),
    (56, 56, 256, 64, 1, 1, 0, 0, 1, 1),
    (56, 56, 256, 128, 1, 1, 0, 0, 2, 2),
    (28, 28, 128, 512, 1, 1, 0, 0, 1, 1),
    (56, 56, 256, 512, 1, 1, 0, 0, 2, 2),
    (28, 28, 512, 128, 1, 1, 0, 0, 1, 1),
    (28, 28, 512, 256, 1, 1, 0, 0, 2, 2),
    (14, 14, 256, 1024, 1, 1, 0, 0, 1, 1),
    (28, 28, 512, 1024, 1, 1, 0, 0, 2, 2),
    (14, 14, 1024, 256, 1, 1, 0, 0, 1, 1),
    (14, 14, 1024, 512, 1, 1, 0, 0, 2, 2),
    (7, 7, 512, 2048, 1, 1, 0, 0, 1, 1),
    (14, 14, 1024, 2048, 1, 1, 0, 0, 2, 2),
    (7, 7, 2048, 512, 1, 1, 0, 0, 1, 1),
]


TARGET_NAME = "llvm -mcpu=skylake-avx512"
NUM_VEC_LANES = 16
DEV = tvm.device(TARGET_NAME, 0)


def get_shape(
    im_height, im_width, in_filter, out_filter, k_h, k_w, hpad, wpad, hstride, wstride, out_dtype
):
    """
    Finds out the shape of all data structures
    """
    ## Find shapes
    data_shape = (1, in_filter // NUM_VEC_LANES, im_height, im_width, NUM_VEC_LANES)

    if out_dtype == "int32":
        kernel_shape = (
            out_filter // NUM_VEC_LANES,
            in_filter // NUM_VEC_LANES,
            k_h,
            k_w,
            NUM_VEC_LANES // 4,
            NUM_VEC_LANES,
            4,
        )
    elif out_dtype == "float32":
        kernel_shape = (
            out_filter // NUM_VEC_LANES,
            in_filter // NUM_VEC_LANES,
            k_h,
            k_w,
            NUM_VEC_LANES,
            NUM_VEC_LANES,
        )
    out_height = (im_height + 2 * hpad - k_h) // hstride + 1
    out_width = (im_width + 2 * wpad - k_w) // wstride + 1
    o_shape = (1, out_filter // NUM_VEC_LANES, out_height, out_width, NUM_VEC_LANES)
    return (data_shape, kernel_shape, o_shape)


def run_inference(
    data_dtype,
    kernel_dtype,
    out_dtype,
    im_height,
    im_width,
    in_filter,
    out_filter,
    k_h,
    k_w,
    hpad,
    wpad,
    hstride,
    wstride,
):
    """
    Runs the inference and checks the functional correctness between
    compute and schedule outputs
    """
    (data_shape, kernel_shape, o_shape) = get_shape(
        im_height,
        im_width,
        in_filter,
        out_filter,
        k_h,
        k_w,
        hpad,
        wpad,
        hstride,
        wstride,
        out_dtype,
    )

    # Create TVM placeholders
    data = te.placeholder(data_shape, name="data", dtype=data_dtype)
    kernel = te.placeholder(kernel_shape, name="kernel", dtype=kernel_dtype)

    # Create the numpy arrays to be used for executing conv models
    if data_dtype == "float32":
        data_array = tvm.nd.array(np.random.rand(*data_shape).astype(dtype=data_dtype), DEV)
        kernel_array = tvm.nd.array(np.random.rand(*kernel_shape).astype(dtype=kernel_dtype), DEV)
    else:
        data_array = tvm.nd.array(np.random.randint(100, size=data_shape).astype(data_dtype))
        kernel_array = tvm.nd.array(np.random.randint(100, size=kernel_shape).astype(kernel_dtype))

    # c_orig will be used for declaration ouptut
    # c_sch will be used for scheduled computation output
    c_orig = tvm.nd.array(np.zeros(o_shape, dtype=out_dtype), DEV)
    c_sch = tvm.nd.array(np.zeros(o_shape, dtype=out_dtype), DEV)

    with tvm.target.Target(TARGET_NAME):
        conv = topi.nn.conv2d_NCHWc(
            data,
            kernel,
            stride=hstride,
            padding=hpad,
            dilation=(1, 1),
            layout="NCHWc",
            out_layout="NCHWc",
            out_dtype=out_dtype,
        )
        out = topi.nn.relu(conv)
        sch = te.create_schedule(out.op)
        func = tvm.build(sch, [data, kernel, out], target=TARGET_NAME, name="out")
        func(data_array, kernel_array, c_orig)
        LOGGER.debug(tvm.lower(sch, [data, kernel], simple_mode=True))

        # Generate and run the optimized schedule
        sconv = topi.generic.nn.schedule_conv2d_NCHWc(outs=[out])
        func = tvm.build(sconv, [data, kernel, out], target=TARGET_NAME, name="conv")
        func(data_array, kernel_array, c_sch)

        # Functional check
        if data_dtype == "uint8":
            np.testing.assert_equal(c_orig.numpy(), c_sch.numpy())
        else:
            assert np.allclose(c_orig.numpy(), c_sch.numpy())

        evaluator = func.time_evaluator(func.entry_name, DEV, number=1000)
        LOGGER.debug(tvm.lower(sconv, [data, kernel], simple_mode=True))
        return evaluator(data_array, kernel_array, c_sch).mean


if __name__ == "__main__":
    LOGGER.info("Workload, Kernel_size, FP32_time, INT8_time, Speedup")
    SPEEDUP_ARRAY = []
    for i, wkl in enumerate(WORKLOADS):
        fp32_time = run_inference("float32", "float32", "float32", *wkl)
        int8_time = run_inference("uint8", "int8", "int32", *wkl)
        kernel_h = wkl[4]
        kernel_w = wkl[5]
        LOGGER.info(
            "Workload#"
            + str(i)
            + ", "
            + str(kernel_h)
            + "x"
            + str(kernel_w)
            + ", "
            + str(fp32_time)
            + ", "
            + str(int8_time)
            + ", "
            + str(fp32_time / int8_time)
        )

        SPEEDUP_ARRAY.append(fp32_time / int8_time)
    LOGGER.info("Average speedup --> %s" % str(sum(SPEEDUP_ARRAY) / float(len(SPEEDUP_ARRAY))))
