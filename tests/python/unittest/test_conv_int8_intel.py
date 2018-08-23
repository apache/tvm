import tvm
import topi
import numpy as np
from tvm.contrib import cc
from tvm.contrib import util
import timeit
from collections import namedtuple

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
ctx = tvm.context(target_name, 0);

def getShape(im_height, im_width, in_filter, out_filter, kh, kw, hpad, wpad,
             hstride, wstride, outDtype):
    ## Find shapes
    dataShape = (1, in_filter/avx2_len, im_height, im_width, avx2_len)

    if outDtype == 'int32':
        if kh != 1:
            kernelShape = (out_filter/avx2_len, in_filter/avx2_len, kh, kw, avx2_len/4, avx2_len, 4)
        else:
            kernelShape = (out_filter/avx2_len, in_filter/avx2_len, avx2_len/4, avx2_len, 4, kh, kw)
    elif outDtype == 'float32':
        if kh != 1:
            kernelShape = (out_filter/avx2_len, in_filter/avx2_len, kh, kw, avx2_len, avx2_len)
        else:
            kernelShape = (out_filter/avx2_len, in_filter/avx2_len, avx2_len, avx2_len, kh, kw)
    out_height = (im_height + 2 * hpad - kh) // hstride + 1
    out_width = (im_width + 2 * wpad - kw) // wstride + 1
    oShape = (1, out_filter/avx2_len, out_height, out_width, avx2_len)
    return (dataShape, kernelShape, oShape)



def run_inference(dataDtype, kernelDtype, outDtype, im_height, im_width, in_filter,
             out_filter, kh, kw, hpad, wpad, hstride, wstride):

    (dataShape, kernelShape, oShape) = getShape(im_height, im_width, in_filter,
                                                out_filter, kh, kw, hpad, wpad,
                                                hstride, wstride, outDtype)

    # Create TVM placeholders
    data = tvm.placeholder(dataShape, name='data', dtype=dataDtype);
    kernel = tvm.placeholder(kernelShape, name='kernel', dtype=kernelDtype);

    # Create the numpy arrays to be used for executing conv models
    if dataDtype == 'float32':
        a = tvm.nd.array(np.random.rand(*dataShape).astype(dtype=dataDtype), ctx);
        b = tvm.nd.array(np.random.rand(*kernelShape).astype(dtype=kernelDtype), ctx);
    else:
        a = tvm.nd.array(np.random.randint(100, size=dataShape).astype(dataDtype));
        b = tvm.nd.array(np.random.randint(100, size=kernelShape).astype(kernelDtype));
        #a = tvm.nd.array(np.ones(dataShape, dtype='uint8'), ctx);
        #b = tvm.nd.array(np.zeros(kernelShape, dtype='int8'), ctx);

    # cOrig will be used for declaration ouptut
    # cSch will be used for scheduled computation output
    cOrig = tvm.nd.array(np.zeros(oShape, dtype=outDtype), ctx);
    cSch = tvm.nd.array(np.zeros(oShape, dtype=outDtype), ctx);


    with tvm.target.create(target_name):
        conv = topi.nn.conv2d_NCHWc(data, kernel, num_filter=out_filter,
                                    kernel_size=(kh, kw), stride=hstride,
                                    padding=hpad, layout='NCHWc',
                                    out_layout='NCHWc', out_dtype=outDtype);
        out = topi.nn.relu(conv)
        s = tvm.create_schedule(out.op);
        func = tvm.build(s, [data, kernel, out], target=target_name, name='out')
        func(a, b, cOrig)
        #print(tvm.lower(s, [data, kernel], simple_mode=True));

        # Generate and run the optimized schedule
        sconv = topi.generic.nn.schedule_conv2d_NCHWc(num_filter=out_filter,
                                                      kernel_size=(kh,kw),
                                                      strides=hstride,
                                                      padding=hpad,
                                                      layout='NCHWc',
                                                      out_layout='NCHWc',
                                                      outs=[out]);
        func = tvm.build(sconv, [data, kernel, out], target=target_name, name='conv')
        func(a, b, cSch)

        # Functional check
        if dataDtype == 'uint8': np.testing.assert_equal(cOrig.asnumpy(), cSch.asnumpy())
        else : assert(np.allclose(cOrig.asnumpy(), cSch.asnumpy()))

        evaluator = func.time_evaluator(func.entry_name, ctx, number=1000)
        #print(tvm.lower(sconv, [data, kernel], simple_mode=True))
        return evaluator(a, b, cSch).mean

if __name__ == "__main__":
    print "Workload, kernelSize, FP32_time, INT8_time, Speedup"
    speedUps = []
    for i in range(0, len(workloads)):
        # workloas[i] -> (im_height, im_width, in_filter, out_filter, kh, kw, hpad, wpad, hstride, wstride)
        # Int8
        fpTime = run_inference('float32','float32','float32', *workloads[i])
        int8Time = run_inference('uint8', 'int8', 'int32', *workloads[i])
        kh = workloads[i][4]
        kw = workloads[i][5]
        print "Workload#" + str(i) + ", " + str(kh) + "x" + str(kw) + ", " + str(fpTime) + ", " + str(int8Time) + ", " + str(fpTime/int8Time)

        speedUps.append(fpTime/int8Time)
    print("Average speedup --> ", sum(speedUps)/float(len(speedUps)))


