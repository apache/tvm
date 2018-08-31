import tvm
import topi
import numpy as np
import timeit
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger('test_conv_int8_intel')
logger.disabled = True

# All the workloads from Resnet except first layer
# Workload is ['height', 'width', 'in_filter', 'out_filter',
#              'hkernel', 'wkernel', 'hpad', 'wpad', 'hstride', 'wstride'])
workloads = [(56, 56, 64, 64, 3, 3, 1, 1, 1, 1),
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
             (7, 7, 2048, 512, 1, 1, 0, 0, 1, 1)
            ]


target_name = 'llvm -mcpu=skylake-avx512'
avx2_len = 16
ctx = tvm.context(target_name, 0)

def get_shape(im_height, im_width, in_filter, out_filter, kh, kw, hpad, wpad,
             hstride, wstride, out_dtype):
    ## Find shapes
    data_shape = (1, in_filter/avx2_len, im_height, im_width, avx2_len)

    if out_dtype == 'int32':
        if kh != 1:
            kernel_shape = (out_filter/avx2_len, in_filter/avx2_len, kh, kw, avx2_len/4, avx2_len, 4)
        else:
            kernel_shape = (out_filter/avx2_len, in_filter/avx2_len, avx2_len/4, avx2_len, 4, kh, kw)
    elif out_dtype == 'float32':
        if kh != 1:
            kernel_shape = (out_filter/avx2_len, in_filter/avx2_len, kh, kw, avx2_len, avx2_len)
        else:
            kernel_shape = (out_filter/avx2_len, in_filter/avx2_len, avx2_len, avx2_len, kh, kw)
    out_height = (im_height + 2 * hpad - kh) // hstride + 1
    out_width = (im_width + 2 * wpad - kw) // wstride + 1
    o_shape = (1, out_filter/avx2_len, out_height, out_width, avx2_len)
    return (data_shape, kernel_shape, o_shape)



def run_inference(data_dtype, kernel_dtype, out_dtype, im_height, im_width, in_filter,
             out_filter, kh, kw, hpad, wpad, hstride, wstride):

    (data_shape, kernel_shape, o_shape) = get_shape(im_height, im_width, in_filter,
                                                out_filter, kh, kw, hpad, wpad,
                                                hstride, wstride, out_dtype)

    # Create TVM placeholders
    data = tvm.placeholder(data_shape, name='data', dtype=data_dtype)
    kernel = tvm.placeholder(kernel_shape, name='kernel', dtype=kernel_dtype)

    # Create the numpy arrays to be used for executing conv models
    if data_dtype == 'float32':
        a = tvm.nd.array(np.random.rand(*data_shape).astype(dtype=data_dtype), ctx)
        b = tvm.nd.array(np.random.rand(*kernel_shape).astype(dtype=kernel_dtype), ctx)
    else:
        a = tvm.nd.array(np.random.randint(100, size=data_shape).astype(data_dtype))
        b = tvm.nd.array(np.random.randint(100, size=kernel_shape).astype(kernel_dtype))

    # cOrig will be used for declaration ouptut
    # cSch will be used for scheduled computation output
    cOrig = tvm.nd.array(np.zeros(o_shape, dtype=out_dtype), ctx)
    cSch = tvm.nd.array(np.zeros(o_shape, dtype=out_dtype), ctx)


    with tvm.target.create(target_name):
        conv = topi.nn.conv2d_NCHWc(data, kernel, num_filter=out_filter,
                                    kernel_size=(kh, kw), stride=hstride,
                                    padding=hpad, layout='NCHWc',
                                    out_layout='NCHWc', out_dtype=out_dtype)
        out = topi.nn.relu(conv)
        s = tvm.create_schedule(out.op)
        func = tvm.build(s, [data, kernel, out], target=target_name, name='out')
        func(a, b, cOrig)
        logger.debug(tvm.lower(s, [data, kernel], simple_mode=True))

        # Generate and run the optimized schedule
        sconv = topi.generic.nn.schedule_conv2d_NCHWc(num_filter=out_filter,
                                                      kernel_size=(kh,kw),
                                                      strides=hstride,
                                                      padding=hpad,
                                                      layout='NCHWc',
                                                      out_layout='NCHWc',
                                                      outs=[out])
        func = tvm.build(sconv, [data, kernel, out], target=target_name, name='conv')
        func(a, b, cSch)

        # Functional check
        if data_dtype == 'uint8': np.testing.assert_equal(cOrig.asnumpy(), cSch.asnumpy())
        else : assert(np.allclose(cOrig.asnumpy(), cSch.asnumpy()))

        evaluator = func.time_evaluator(func.entry_name, ctx, number=1000)
        logger.debug(tvm.lower(sconv, [data, kernel], simple_mode=True))
        return evaluator(a, b, cSch).mean

if __name__ == "__main__":
    logger.info("Workload, kernelSize, FP32_time, INT8_time, Speedup")
    speedUps = []
    for i in range(0, len(workloads)):
        # workloas[i] -> (im_height, im_width, in_filter, out_filter, kh, kw, hpad, wpad, hstride, wstride)
        # Int8
        fpTime = run_inference('float32','float32','float32', *workloads[i])
        int8Time = run_inference('uint8', 'int8', 'int32', *workloads[i])
        kh = workloads[i][4]
        kw = workloads[i][5]
        logger.info("Workload#" + str(i) + ", " + str(kh) + "x" + str(kw) + ", " + str(fpTime) + ", " + str(int8Time) + ", " + str(fpTime/int8Time))

        speedUps.append(fpTime/int8Time)
    logger.info("Average speedup --> ", sum(speedUps)/float(len(speedUps)))


