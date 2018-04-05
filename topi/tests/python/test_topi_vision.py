"""Test code for vision package"""
import numpy as np
import tvm
import topi
import math

from topi.vision import ssd


def verify_multibox_prior(dshape, sizes=(1,), ratios=(1,), steps=(-1, -1), offsets=(0.5, 0.5), clip=False):
    data = tvm.placeholder(dshape, name="data")
    out = ssd.multibox_prior(data, sizes, ratios, steps, offsets, clip)

    dtype = data.dtype
    input_data = np.random.uniform(size=dshape).astype(dtype)

    in_height = data.shape[2].value
    in_width = data.shape[3].value
    num_sizes = len(sizes)
    num_ratios = len(ratios)
    size_ratio_concat = sizes + ratios
    steps_h = steps[0] if steps[0] > 0 else 1.0 / in_height
    steps_w = steps[1] if steps[1] > 0 else 1.0 / in_width
    offset_h = offsets[0]
    offset_w = offsets[1]

    oshape = (1, in_height * in_width * (num_sizes + num_ratios - 1), 4)
    np_out = np.zeros(oshape).astype(dtype)

    for i in range(in_height):
        center_h = (i + offset_h) * steps_h
        for j in range(in_width):
            center_w = (j + offset_w) * steps_w
            for k in range(num_sizes + num_ratios - 1):
                w = size_ratio_concat[k] * in_height / in_width / 2.0 if k < num_sizes else \
                    size_ratio_concat[0] * in_height / in_width * math.sqrt(size_ratio_concat[k + 1]) / 2.0
                h = size_ratio_concat[k] / 2.0 if k < num_sizes else \
                    size_ratio_concat[0] / math.sqrt(size_ratio_concat[k + 1]) / 2.0
                count = i * in_width * (num_sizes + num_ratios - 1) + j * (num_sizes + num_ratios - 1) + k
                np_out[0][count][0] = center_w - w
                np_out[0][count][1] = center_h - h
                np_out[0][count][2] = center_w + w
                np_out[0][count][3] = center_h + h
    if clip:
        np_out = np.clip(np_out, 0, 1)

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s = topi.generic.schedule_multibox_prior(out)

        tvm_input_data = tvm.nd.array(input_data, ctx)
        tvm_out = tvm.nd.array(np.zeros(oshape, dtype=dtype), ctx)
        f = tvm.build(s, [data, out], device)
        f(tvm_input_data, tvm_out)
        np.testing.assert_allclose(tvm_out.asnumpy(), np_out, rtol=1e-4)

    for device in ['llvm', 'cuda']:
        check_device(device)


def test_multibox_prior():
    verify_multibox_prior((1, 3, 50, 50))
    verify_multibox_prior((1, 3, 224, 224), sizes=(0.5, 0.25, 0.1), ratios=(1, 2, 0.5))
    verify_multibox_prior((1, 32, 32, 32), sizes=(0.5, 0.25), ratios=(1, 2), steps=(2, 2), clip=True)


if __name__ == "__main__":
    test_multibox_prior()