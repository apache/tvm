"""Test code for vision package"""
import numpy as np
import tvm
import topi
import math

from topi.vision import ssd, nms


def test_nms():
    dshape = (1, 5, 6)
    data = tvm.placeholder(dshape, name="data")
    valid_count = tvm.placeholder((dshape[0],), dtype="int32", name="valid_count")
    nms_threshold = 0.7
    force_suppress = True
    nms_topk = 2

    np_data = np.array([[[0, 0.8, 1, 20, 25, 45], [1, 0.7, 30, 60, 50, 80],
                         [0, 0.4, 4, 21, 19, 40], [2, 0.9, 35, 61, 52, 79],
                         [1, 0.5, 100, 60, 70, 110]]]).astype(data.dtype)
    np_valid_count = np.array([4]).astype(valid_count.dtype)
    np_result = np.array([[[2, 0.9, 35, 61, 52, 79], [0, 0.8, 1, 20, 25, 45],
                           [0, 0.4, 4, 21, 19, 40], [-1, 0.9, 35, 61, 52, 79],
                           [-1, -1, -1, -1, -1, -1]]])

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            if device == 'llvm':
                out = nms(data, valid_count, nms_threshold, force_suppress, nms_topk)
            else:
                out = topi.cuda.nms(data, valid_count, nms_threshold, force_suppress, nms_topk)
            s = topi.generic.schedule_nms(out)

        tvm_data = tvm.nd.array(np_data, ctx)
        tvm_valid_count = tvm.nd.array(np_valid_count, ctx)
        tvm_out = tvm.nd.array(np.zeros(dshape, dtype=data.dtype), ctx)
        f = tvm.build(s, [data, valid_count, out], device)
        f(tvm_data, tvm_valid_count, tvm_out)
        np.testing.assert_allclose(tvm_out.asnumpy(), np_result, rtol=1e-4)

    for device in ['llvm', 'opencl']:
        check_device(device)


def verify_multibox_prior(dshape, sizes=(1,), ratios=(1,), steps=(-1, -1), offsets=(0.5, 0.5), clip=False):
    data = tvm.placeholder(dshape, name="data")

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
            if device == 'llvm':
                out = ssd.multibox_prior(data, sizes, ratios, steps, offsets, clip)
            else:
                out = topi.cuda.ssd.multibox_prior(data, sizes, ratios, steps, offsets, clip)
            s = topi.generic.schedule_multibox_prior(out)

        tvm_input_data = tvm.nd.array(input_data, ctx)
        tvm_out = tvm.nd.array(np.zeros(oshape, dtype=dtype), ctx)
        f = tvm.build(s, [data, out], device)
        f(tvm_input_data, tvm_out)
        np.testing.assert_allclose(tvm_out.asnumpy(), np_out, rtol=1e-3)

    for device in ['llvm', 'opencl']:
        check_device(device)


def test_multibox_prior():
    verify_multibox_prior((1, 3, 50, 50))
    verify_multibox_prior((1, 3, 224, 224), sizes=(0.5, 0.25, 0.1), ratios=(1, 2, 0.5))
    verify_multibox_prior((1, 32, 32, 32), sizes=(0.5, 0.25), ratios=(1, 2), steps=(2, 2), clip=True)


def test_multibox_detection():
    batch_size = 1
    num_anchors = 3
    num_classes = 3
    cls_prob = tvm.placeholder((batch_size, num_anchors, num_classes), name="cls_prob")
    loc_preds = tvm.placeholder((batch_size, num_anchors * 4), name="loc_preds")
    anchors = tvm.placeholder((1, num_anchors, 4), name="anchors")

    # Manually create test case
    np_cls_prob = np.array([[[0.2, 0.5, 0.3], [0.25, 0.3, 0.45], [0.7, 0.1, 0.2]]])
    np_loc_preds = np.array([[0.1, -0.2, 0.3, 0.2, 0.2, 0.4, 0.5, -0.3, 0.7, -0.2, -0.4, -0.8]])
    np_anchors = np.array([[[-0.1, -0.1, 0.1, 0.1], [-0.2, -0.2, 0.2, 0.2], [1.2, 1.2, 1.5, 1.5]]])

    expected_np_out = np.array([[[1, 0.69999999, 0, 0, 0.10818365, 0.10008108],
                                 [0, 0.44999999, 1, 1, 1, 1],
                                 [0, 0.30000001, 0, 0, 0.22903419, 0.20435292]]])

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            if device == 'llvm':
                out = ssd.multibox_detection(cls_prob, loc_preds, anchors)
            else:
                out = topi.cuda.ssd.multibox_detection(cls_prob, loc_preds, anchors)
            s = topi.generic.schedule_multibox_detection(out)

        tvm_cls_prob = tvm.nd.array(np_cls_prob.astype(cls_prob.dtype), ctx)
        tvm_loc_preds = tvm.nd.array(np_loc_preds.astype(loc_preds.dtype), ctx)
        tvm_anchors = tvm.nd.array(np_anchors.astype(anchors.dtype), ctx)
        tvm_out = tvm.nd.array(np.zeros((batch_size, num_anchors, 6)).astype(out.dtype), ctx)
        f = tvm.build(s, [cls_prob, loc_preds, anchors, out], device)
        f(tvm_cls_prob, tvm_loc_preds, tvm_anchors, tvm_out)
        np.testing.assert_allclose(tvm_out.asnumpy(), expected_np_out, rtol=1e-4)

    for device in ['llvm', 'opencl']:
        check_device(device)


if __name__ == "__main__":
    test_nms()
    test_multibox_prior()
    test_multibox_detection()
