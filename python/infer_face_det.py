import argparse
import os
import numpy as np
import tvm
import tvm.contrib.graph_executor as runtime
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
import cv2
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms
# import onnx
# import onnxruntime as ort
import time


# from keras.applications.inception_resnet_v2 import preprocess_input


def make_parser():
    parser = argparse.ArgumentParser("trt_onnx")
    parser.add_argument("-p", "--path", type=str, default=None)
    parser.add_argument("-rp", "--rec_path", type=str, default=None)
    parser.add_argument("-d", "--device", type=str, default='cuda')
    parser.add_argument("-ep", "--export_path", type=str, default="./relays")
    parser.add_argument("--input_size", nargs='+', type=int, default=[1, 3, 112, 112],
                        help="input size in list")
    parser.add_argument("--input_name", type=str, default="images",
                        help="input node name")
    parser.add_argument("--input_img", type=str, default="random",
                        help="input data from image or random generated")
    parser.add_argument("--face_lib_dir", type=str, default=None)
    return parser


def similarTransfor(src: np.ndarray, dst: np.ndarray):
    Hmat = cv2.findHomography(src, dst)
    return Hmat[0]
    # src_mean = np.sum(src, axis=0) / src.shape[0]
    # dst_mean = np.sum(dst, axis=0) / dst.shape[0]
    # src = src - src_mean
    # dst = src - dst_mean
    # print(src)
    # print(dst)
    # A = np.dot(np.transpose(dst), src) / dst.shape[0]
    # d = np.ones(shape=(src.ndim, 1), dtype=np.float32)
    # if cv2.determinant(A) < 0:
    #     d[src.ndim-1, 0] = -1
    # T = np.eye(src.ndim+1, src.ndim+1, dtype=np.float32)
    # S, U, V = cv2.SVDecomp(A)


def face_alignment(ori_img: np.ndarray, det: np.ndarray, cfg):
    print(det.shape)
    anchors = np.array(det[5:], dtype=np.float32)
    v2 = anchors.reshape(5, 2)
    v1 = np.array(cfg["land_mask_v1"], dtype=np.float32)
    hmat = similarTransfor(v2, v1)
    aligned = cv2.warpPerspective(ori_img, M=hmat, dsize=(96, 112), flags=cv2.INTER_CUBIC)
    cv2.imwrite("./aligned.jpg", aligned)
    return aligned


def load_graph_exec(device, _target, fname):
    module_path = os.path.join("./", fname)
    if device == "arm_cuda":
        remote = autotvm.measure.request_remote("tx2", "192.168.6.252",
                                                9190, timeout=10000)
        remote.upload(module_path)
        rlib = remote.load_module(fname)
        rdev = remote.device(str(_target), 0)
        rmodule = runtime.GraphModule(rlib["default"](rdev))
        return rmodule, rdev
    else:
        lib = tvm.runtime.load_module(module_path)
        dev = tvm.device((str(_target)), 0)
        module = runtime.GraphModule(lib["default"](dev))
        return module, dev


def face_detect(face_img, cfg, priors, mod, dev):
    dtype = "float32"
    scale = np.array([face_img.shape[1], face_img.shape[0], face_img.shape[1], face_img.shape[0]])
    scale1 = np.array([face_img.shape[1], face_img.shape[0], face_img.shape[1], face_img.shape[0],
                       face_img.shape[1], face_img.shape[0], face_img.shape[1], face_img.shape[0],
                       face_img.shape[1], face_img.shape[0]])
    face_img = cv2.resize(face_img, (640, 480))
    face_img = np.transpose(face_img, (2, 0, 1))
    face_img = face_img.astype(dtype)
    face_img = np.expand_dims(face_img, 0)
    data_tvm = tvm.nd.array(face_img, dev)
    mod.set_input("input", data_tvm)
    mod.run()
    loc = mod.get_output(0).numpy()
    conf = mod.get_output(1).numpy()
    landms = mod.get_output(2).numpy()
    boxes = decode(np.array(loc.squeeze(0)), priors,
                   cfg["variance"])
    boxes = boxes * scale
    scores = conf.squeeze(0)[:, 1]
    landms = decode_landm(np.array(landms.squeeze(0)),
                          priors,
                          cfg["variance"])

    landms = landms * scale1
    # ignore low scores
    inds = np.where(scores > 0.5)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    top_k = 1000
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    nms_threshold = 0.4
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    keep_top_k = 750
    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)
    return dets


def construct_face_feat_lib(path_dir, cfg, prior, mod_det, mod, device):
    images = os.listdir(path_dir)
    face_feats = []
    face_names = []
    for im_path in images:
        print(im_path)
        face_path = os.path.join(path_dir, im_path)
        im_origin = cv2.imread(face_path)
        # do detection first
        face_im = np.float32(im_origin)
        dets = face_detect(face_im, cfg, prior, mod_det, device)
        print("detected : " + str(len(dets)))
        # do face alignment
        assert len(dets) == 1
        aligned = face_alignment(np.float32(im_origin), dets[0], cfg)
        im = np.float32(aligned)

        im = cv2.resize(im, (112, 112))
        im = im.astype("float32")
        im = im / 127.5
        im = im - 1.0
        im = np.transpose(im, (2, 0, 1))

        im = np.expand_dims(im, 0)
        im = im.astype("float32")
        dv = tvm.nd.array(im, device)

        tic = time.time()
        mod.set_input("input.1", dv)
        mod.run()
        rv = mod.get_output(0).numpy()
        print('net forward time: {:.4f}'.format((time.time() - tic) / 1.0))
        rv = np.array(rv).squeeze(0)
        face_feats.append(rv)
        face_names.append(im_path)
        # ort_sess = ort.InferenceSession("D:\\workspace\\project\\nn_compiler\\face\\facenet-38.onnx")
        # outputs = ort_sess.run(None, {'input.1': im})
        # a = 0
        # tic = time.time()
        # for i in range(0, 100):
        #     outputs = ort_sess.run(None, {'input.1': im})
        # print('net forward time: {:.4f}'.format((time.time() - tic)/100.0))

    return face_names, face_feats


def main(args):
    target = args.device
    cfg = cfg_mnet
    module, dev = load_graph_exec(args.device, target, args.path)
    # module_rec, dev_rec = load_graph_exec("llvm", "llvm", args.rec_path)
    module_rec, dev_rec = load_graph_exec(args.device, target, args.rec_path)
    priorbox = PriorBox(cfg, image_size=(480, 640))
    priors = priorbox.forward()
    face_names, face_feats = construct_face_feat_lib(args.face_lib_dir, cfg, priors,
                                                     module, module_rec, dev_rec)

    # load image from cv2
    print("finish warm up , begin to inference...")
    dtype = "float32"
    img_raw = cv2.imread(args.input_img)
    face_img_origin = np.float32(img_raw)
    dets = face_detect(img_raw, cfg, priors, module, dev)

    for det in dets:

        if det[4] < 0.6:
            continue
        text = "{:.4f}".format(det[4])
        b = list(map(int, det))
        cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        face_2_rec = face_img_origin[b[1]:b[3], b[0]:b[2], :]
        aligned = face_alignment(face_img_origin, det, cfg)
        face_2_rec = cv2.resize(aligned, (112, 112))
        im = face_2_rec
        # cv2.imwrite("test_rec.jpg", face_2_rec)

        im = im.astype("float32")
        im = im / 127.5
        im = im - 1.0
        im = np.transpose(im, (2, 0, 1))

        im = np.expand_dims(im, 0)
        im = im.astype("float32")
        data_rec_tvm = tvm.nd.array(im, dev)
        module_rec.set_input("input.1", data_rec_tvm)
        module_rec.run()
        data_feat_tvm = module_rec.get_output(0)
        data_feat = data_feat_tvm.numpy().squeeze(0)
        dist_data_feat = np.sqrt(np.sum(np.square(data_feat)))
        data_feat = data_feat / dist_data_feat
        max_val = 0
        max_id = -1
        for i in range(0, len(face_names)):
            feat = face_feats[i]
            dist_feat = np.sqrt(np.sum(np.square(feat)))
            feat = feat / dist_feat
            correctness = np.sum(data_feat * feat)
            if correctness >= max_val:
                max_val = correctness
                max_id = face_names[i]
            print(face_names[i] + " : " + str(correctness))
        if max_val > 0.15:
            cv2.putText(img_raw, max_id, (b[0], b[1] + 24), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        else:
            cv2.putText(img_raw, "unknown", (b[0], b[1] + 24), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        print("---------------------------------------")

        cx = b[0]
        cy = b[1] + 12
        cv2.putText(img_raw, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # landms
        cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
        cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
        cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
        cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
        cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)

    name = "test.jpg"
    cv2.imwrite(name, img_raw)

    exit(0)


if __name__ == '__main__':
    args = make_parser().parse_args()
    main(args)
