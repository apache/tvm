import nnvm.symbol as symbol
import nnvm.graph as graph
import nnvm.compiler.graph_util as graph_util
import nnvm.compiler
import numpy as np, numpy.testing as npt
import zipfile
import os
import time
import tvm
from nnvm.testing import utils
from tvm.contrib import graph_runtime, util
import mxnet as mx
import cv2
from nnvm.frontend import from_mxnet
from tvm.contrib.download import download
from mxnet.model import load_checkpoint


def test_graph_annotation():
    def execute_original_graph(sym, target=None, shape=None, dtype="float32",
                               params=None, target_host=None, layout=None):
        subgraph = graph.create(sym)
        deploy_graph, lib, params = nnvm.compiler.build(
            subgraph, target=target, shape=shape, dtype=dtype, params=params,
            target_host=target_host, layout=layout)

        ctx = tvm.cpu()
        module = graph_runtime.create(deploy_graph, lib, ctx)
        module.set_input(**params)
        module.run()
        _, oshape = graph_util.infer_shape(deploy_graph)
        module_out = []
        for i in range(len(sym.list_output_names())):
            out = module.get_output(i, out=tvm.nd.empty(oshape[i], dtype))
            module_out.append(out)
        return module_out

    def check_annotated_graph(sym, op_names, expected_num_nodes,
                              data_shape=None, params=None):
        targets = {"cpu": "llvm", "opencl": "opencl"}

        deploy_graph, lib_dev, params = nnvm.compiler.build_heterogeneous(
            sym, targets=targets, shape=data_shape, dtype="float32",
            params=params, op_names=op_names)

        new_sym = deploy_graph.symbol()
        assert len(new_sym.list_input_names()) == len(sym.list_input_names())
        assert len(new_sym.list_output_names()) == len(sym.list_output_names())
        assert deploy_graph.index.num_nodes == expected_num_nodes

    def test_conv_network():
        """ The network is as following:
                data1       data2
                  |           |
                conv2d      conv2d
                   \         /
                  elemwise_add
                        |
                      conv2d
        """
        out_channels = 16
        data1 = symbol.Variable(name="data1")
        data2 = symbol.Variable(name="data2")
        simple_net1 = symbol.conv2d(data=data1, kernel_size=(3, 3),
                                    channels=out_channels, padding=(1, 1),
                                    use_bias=True)

        simple_net2 = symbol.conv2d(data=data2, kernel_size=(3, 3),
                                    channels=out_channels, padding=(1, 1),
                                    use_bias=True)
        ret = symbol.elemwise_add(simple_net1, simple_net2)
        ret = symbol.conv2d(ret, kernel_size=(3, 3),
                            channels=out_channels, padding=(1, 1),
                            use_bias=True)

        batch_size = 1
        data_shape = (batch_size, 3, 224, 224)
        shape_dict = {"data1": data_shape, "data2": data_shape}
        params = {}
        params["data1"] = np.random.uniform(-1, 1,
                                            size=data_shape).astype("float32")
        params["data2"] = np.random.uniform(-1, 1,
                                            size=data_shape).astype("float32")
        # No op will be fused. 3 additional device copy nodes are required.
        check_annotated_graph(ret, ["elemwise_add"], 15, shape_dict, params)

    def test_fusible_network():
        """ The network is as following:
                    data
                      |
                     exp
                    /   \
                 sqrt   log
                    \   /
                    b_add
                      |
                    tanh
        """
        batch_size = 1
        data_shape = (batch_size, 3, 224, 224)
        data = symbol.Variable('data', shape=data_shape, dtype="float32")
        shape_dict = {"data": data_shape}
        params = {}
        params["data"] = np.random.uniform(-1, 1,
                                           size=data_shape).astype("float32")

        exp = symbol.exp(data, name='exp')
        sqrt = symbol.sqrt(exp, name='sqrt')
        log = symbol.log(exp, name='log')
        ret = sqrt + log
        ret = symbol.tanh(ret)

        # Fuse log and broadcast_add.
        check_annotated_graph(ret, ['exp', 'log', 'broadcast_add'], 8,
                              shape_dict,
                              params)

        # Fuse log, broadcast_add, and tanh
        check_annotated_graph(ret, ['exp', 'sqrt', 'none', 'elemwise_add'], 6,
                              shape_dict, params)

        # No operator will be fused.
        check_annotated_graph(ret, ['log', 'sqrt', 'none', 'tanh'], 11,
                              shape_dict, params)

        # All operators will be fused.
        check_annotated_graph(ret, [''], 2, shape_dict, params)

        # All operators will be fused since all of them are annotated to the
        # same device.
        check_annotated_graph(ret,
                              ['exp', 'sqrt', 'broadcast_add', 'none', 'log',
                               'tanh'], 2, shape_dict, params)

        # Fuse exp, sqrt, log, and boradcast_add
        check_annotated_graph(ret, ['tanh'], 4, shape_dict, params)

    def check_graph(sym, op_names, data_shape, params):
        dtype = "float32"
        targets = {"cpu": "llvm", "opencl": "opencl"}

        # execute the whole graph on cpu
        shape1 = {k: v for k, v in data_shape.items()}
        params1 = {k: tvm.nd.array(v) for k, v in params.items()}
        orig_out = execute_original_graph(sym, target="llvm", shape=shape1,
                                          dtype=dtype, params=params1)

        # annotate and compile the graph
        deploy_graph, lib_dev, params = nnvm.compiler.build_heterogeneous(
            sym, targets=targets, shape=data_shape, dtype=dtype, params=params,
            op_names=op_names)

        module = graph_runtime.create(deploy_graph, lib_dev, tvm.context("cpu"))
        module.set_input(**params)
        module.run()
        _, oshape = graph_util.infer_shape(deploy_graph)
        module_out = []
        for i in range(len(sym.list_output_names())):
            out = module.get_output(i, out=tvm.nd.empty(oshape[i], dtype))
            module_out.append(out)
            npt.assert_allclose(out.asnumpy(), orig_out[i].asnumpy(),
                                rtol=1e-5, atol=1e-5)

    def test_duplex_data_transfer():
        """ This unittest tests duplex communication between the host and
        accelerator device. The network is as following:
                    data
                      |
                    conv2d  (acc)
                      |
                 batch_norm (cpu)
                      |
                    conv2d  (acc)
        """
        out_channels = 16
        data = symbol.Variable(name="data")
        simple_net = symbol.conv2d(data=data, kernel_size=(3, 3),
                                   channels=out_channels, padding=(1, 1),
                                   use_bias=False)
        simple_net = symbol.batch_norm(simple_net)
        simple_net = symbol.conv2d(data=simple_net, kernel_size=(3, 3),
                                   channels=out_channels, padding=(1, 1),
                                   use_bias=False)

        batch_size = 1
        data_shape = (batch_size, 3, 224, 224)
        shape_dict = {"data": data_shape}
        net, params = utils.create_workload(simple_net, batch_size,
                                            data_shape[1:])
        params["data"] = data = np.random.uniform(-1, 1,
                                                  size=data_shape).astype(
            "float32")

        check_graph(net, ['batch_norm'], shape_dict, params)

    def heterogeneous_ssd(sym, op_names, data_shape=None, params=None,
                          test_image_path=None):
        target, dtype = "llvm", "float32"

        # targets = {"cpu": "llvm", "opencl": str(
        #    tvm.target.intel_graphics())}
        targets = {"cpu": "llvm", "opencl": "opencl"}

        with nnvm.compiler.build_config(opt_level = 3):
            deploy_graph, lib_dev, params = \
                nnvm.compiler.build_heterogeneous(sym, targets=targets,
                                                  shape=data_shape,
                                                  dtype=dtype, params=params,
                                                  op_names=op_names)
            # import sys
            # sys.stdout = open("annotated.json", "w")
            # print(deploy_graph.json)

        host_ctx = tvm.context("cpu")
        module = graph_runtime.create(deploy_graph, lib_dev, host_ctx)

        dshape = data_shape["data"]
        # Preprocess image
        image = cv2.imread(test_image_path)
        img_data = cv2.resize(image, (dshape[2], dshape[3]))
        img_data = img_data[:, :, (2, 1, 0)].astype(np.float32)
        img_data -= np.array([123, 117, 104])
        img_data = np.transpose(np.array(img_data), (2, 0, 1))
        img_data = np.expand_dims(img_data, axis=0)

        module.set_input('data', tvm.nd.array(img_data.astype(dtype)))
        module.set_input(**params)
        module.run()
        _, oshape = graph_util.infer_shape(
            deploy_graph, shape={"data": dshape})
        tvm_output = module.get_output(
            0, tvm.nd.empty(tuple(oshape[0]), dtype))
        ftimer = module.module.time_evaluator("run", host_ctx, 2)
        for i in range(2):
            prof_res = ftimer()
            print(prof_res)
            # sleep for avoiding device overheat
            if i + 1 != 5:
                time.sleep(45)

        return image, tvm_output

    # test sdd
    def test_ssd():
        model_name = "ssd_resnet50_512"
        model_file = "%s.zip" % model_name
        test_image = "dog.jpg"
        dshape = (1, 3, 512, 512)

        ######################################################################
        # Download MXNet SSD pre-trained model and demo image
        # ---------------------------------------------------
        # Pre-trained model available at
        # https://github.com/apache/incubator-\mxnet/tree/master/example/ssd

        model_url = "https://github.com/zhreshold/mxnet-ssd/releases/download/v0.6/" \
                    "resnet50_ssd_512_voc0712_trainval.zip"
        image_url = "https://cloud.githubusercontent.com/assets/3307514/20012567/" \
                    "cbb60336-a27d-11e6-93ff-cbc3f09f5c9e.jpg"
        inference_symbol_folder = "c1904e900848df4548ce5dfb18c719c7-a28c4856c827fe766aa3da0e35bad41d44f0fb26"
        inference_symbol_url = "https://gist.github.com/kevinthesun/c1904e900848df4548ce5dfb18c719c7/" \
                               "archive/a28c4856c827fe766aa3da0e35bad41d44f0fb26.zip"

        dir = "ssd_model"
        if not os.path.exists(dir):
            os.makedirs(dir)
        model_file_path = "%s/%s" % (dir, model_file)
        test_image_path = "%s/%s" % (dir, test_image)
        inference_symbol_path = "%s/inference_model.zip" % dir
        download(model_url, model_file_path)
        download(image_url, test_image_path)
        download(inference_symbol_url, inference_symbol_path)

        zip_ref = zipfile.ZipFile(model_file_path, 'r')
        zip_ref.extractall(dir)
        zip_ref.close()
        zip_ref = zipfile.ZipFile(inference_symbol_path)
        zip_ref.extractall(dir)
        zip_ref.close()

        ######################################################################
        # Convert and compile model with NNVM for CPU.
        sym = mx.sym.load("%s/%s/ssd_resnet50_inference.json" %
                          (dir, inference_symbol_folder))
        _, arg_params, aux_params = load_checkpoint(
            "%s/%s" % (dir, model_name), 0)
        net, params = from_mxnet(sym, arg_params, aux_params)

        shape_dict = {"data": dshape}
        with nnvm.compiler.build_config(opt_level=3):
            image, tvm_output = heterogeneous_ssd(net, ['nms'],
                                                  shape_dict,
                                                  params, test_image_path)

        #####################################################################

        # Display result

        class_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
                       "car", "cat", "chair",
                       "cow", "diningtable", "dog", "horse", "motorbike",
                       "person", "pottedplant",
                       "sheep", "sofa", "train", "tvmonitor"]

        def display(img, out, thresh=0.5):
            import random
            import matplotlib as mpl
            import matplotlib.pyplot as plt
            mpl.rcParams['figure.figsize'] = (10, 10)
            pens = dict()
            plt.clf()
            plt.imshow(img)
            for det in out:
                cid = int(det[0])
                if cid < 0:
                    continue
                score = det[1]
                if score < thresh:
                    continue
                if cid not in pens:
                    pens[cid] = (random.random(),
                                 random.random(), random.random())
                scales = [img.shape[1], img.shape[0]] * 2
                xmin, ymin, xmax, ymax = [
                    int(p * s) for p, s in zip(det[2:6].tolist(), scales)]
                rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     fill=False,
                                     edgecolor=pens[cid], linewidth=3)
                plt.gca().add_patch(rect)
                text = class_names[cid]
                plt.gca().text(xmin, ymin - 2,
                               '{:s} {:.3f}'.format(text, score),
                               bbox=dict(facecolor=pens[cid], alpha=0.5),
                               fontsize=12, color='white')
            plt.show()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        display(image, tvm_output.asnumpy()[0], thresh=0.45)

    # These tests are performed using OpenCL as the default device. The
    # specified operators are scheduled to CPU.
    test_conv_network()
    test_fusible_network()
    test_duplex_data_transfer()
    test_ssd()


if __name__ == "__main__":
    test_graph_annotation()
