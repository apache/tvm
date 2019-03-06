import logging
import argparse
import os
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.model_zoo import vision
import numpy as np

# Two functions for reading data from record file or raw images
def get_val_data(args,
                 rec_val,
                 batch_size,
                 num_workers=4):
    rec_val = os.path.expanduser(rec_val)
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]
    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        return data, label

    img_size = 299 if args.model == 'inceptionv3' else 224
    val_data = mx.io.ImageRecordIter(
        path_imgrec         = rec_val,
        preprocess_threads  = num_workers,
        shuffle             = False,
        batch_size          = batch_size,
        resize              = 256,
        data_shape          = (3, img_size, img_size),
        mean_r              = mean_rgb[0],
        mean_g              = mean_rgb[1],
        mean_b              = mean_rgb[2],
        std_r               = std_rgb[0],
        std_g               = std_rgb[1],
        std_b               = std_rgb[2],
    )
    return val_data, batch_fn


def evaluate(args, graph, lib, params, ctx, free_vars):
#def evaluate(args, graph, intrp, params):
    """Evaluate on the validation set."""
    import tvm
    from tvm.contrib import graph_runtime

    # tetup dataset.
    batch_size = args.batch_size
    val_data, batch_fn = get_val_data(args, args.rec_val, batch_size)
    # create runtime module
    m = graph_runtime.create(graph, lib, ctx)
    scales = {}
    print(len(free_vars))
    for free_var in free_vars:
        print(free_var.name_hint)
        params[str(free_var.name_hint)] = np.array(8.0/128)
        #scales[str(free_var.name_hint)] = np.array(8.0/128)

    m.set_input(**params)
    #m.set_input(**scales)
    oshape = (batch_size, args.num_classes)
    out_arr = tvm.nd.empty(oshape, "float32")
    # setup evaluaiton metric
    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)
    val_data.reset()
    acc_top1.reset()
    acc_top5.reset()
    # Execute
    for i, batch in enumerate(val_data):
        data, label = batch_fn(batch, [mx.cpu(0)])
        m.run(data=data[0].asnumpy())
        m.run(data=data[0].asnumpy(), **scales)
        m.get_output(0, out_arr)
        #out_arr = intrp.evaluate(graph)(data)
        acc_top1.update(label, [mx.nd.array(out_arr.asnumpy())])
        acc_top5.update(label, [mx.nd.array(out_arr.asnumpy())])

        if args.log_interval and not (i + 1) % args.log_interval:
            _, top1 = acc_top1.get()
            _, top5 = acc_top5.get()
            nsamples = (i + 1) * batch_size
            logging.info('[%d samples] validation: acc-top1=%f acc-top5=%f', nsamples, top1, top5)
    logging.info('[final] validation: acc-top1=%f acc-top5=%f', top1, top5)
    with open('record.csv', "a") as f:
        f.write('{0}, {1}, {2}, {3}, {4}\n'.format(
            args.model, args.nbit_input, args.nbit_output, args.global_scale, top1))


def build_model(args, gluon_model):
    """Build with relay."""
    import tvm
    from tvm import relay
    from tvm.relay import quantize as qtz
    img_size = 299 if args.model == 'inceptionv3' else 224
    data_shape = (args.batch_size, 3, img_size, img_size)
    net, params = relay.frontend.from_mxnet(gluon_model, {"data": data_shape})
    target = args.target

    if args.original:
        # run original model
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(net, target, params=params)
        ctx = tvm.nd.context(target, 0)
        return graph, lib, params, ctx

    # constant folding and scale folding.
    #print('original')
    #print(net.astext(show_meta_data=False))
    with relay.build_config(opt_level=3):
        qgraph = relay.optimize(net, target, params)
        # qgraph = relay.optimize(qgraph)
    #print('after optimize')
    #print(qgraph.astext(show_meta_data=False))

    with qtz.qconfig(skip_k_conv=0,
                     nbit_input=args.nbit_input,
                     nbit_weight=args.nbit_input,
                     global_scale=args.global_scale,
                     dtype_input=args.dtype_input,
                     dtype_weight=args.dtype_input,
                     dtype_activation=args.dtype_output,
                     store_lowbit_output=False,
                     debug_enabled_ops=None):
        print(qtz.current_qconfig())
        qgraph = qtz.annotate(qgraph)
        print('after annotate')
        print(qgraph.astext(show_meta_data=False))
        qgraph = qtz.calibrate(qgraph)
        free_vars = []
        free_vars = list(relay.ir_pass.free_vars(qgraph))
        qgraph = relay.Function(list(qgraph.params) + free_vars,
                                qgraph.body, qgraph.ret_type,
                                qgraph.type_params, qgraph.attrs)
        print('after calibrate\n')
        print(qgraph.astext(show_meta_data=False))
        if not args.simulated:
            qgraph = qtz.realize(qgraph)
            qgraph = relay.ir_pass.infer_type(qgraph)
            print('after realize\n')
            print(qgraph.astext(show_meta_data=False))

    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(qgraph, target)
        ctx = tvm.nd.context(target, 0)
        #intrp = relay.build_module.create_executor('graph', qgraph, ctx, target)
    return graph, lib, params, ctx, free_vars
    #return qgraph, intrp, params


def main(args):
    gluon_model = vision.get_model(args.model, pretrained=True)
    graph, lib, params, ctx, free_vars = build_model(args, gluon_model)
    #graph, intrp, params = build_model(args, gluon_model)
    logging.info("Finish building model %s...", args.model)
    # raise ValueError
    evaluate(args, graph, lib, params, ctx, free_vars)
    #evaluate(args, graph, intrp, params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ImageNet validation accuracy")
    parser.add_argument("--rec-val", type=str, default="/scratch/eqy/imagenet/val.rec",
                        help="the validation data")
    parser.add_argument("--num-classes", type=int, default=1000,
                        help="batch size")
    parser.add_argument("--model", type=str, default="resnet18_v1",
                        help="Name of the model")
    parser.add_argument("--log-interval", type=int, default=100,
                        help="log interval")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch size")
    parser.add_argument("--target", type=str, default="llvm",
                        help="target option")
    parser.add_argument("--nbit-input", type=int, default=8,
                        help="number of input bits")
    parser.add_argument("--nbit-output", type=int, default=32,
                        help="number of output bits")
    parser.add_argument("--dtype-input", type=str, default="int8",
                        help="number of input bits")
    parser.add_argument("--dtype-output", type=str, default="int32",
                        help="number of output bits")
    parser.add_argument("--global-scale", type=float, default=8.0,
                        help="global activation scale")
    parser.add_argument("--original", action="store_true",
                        help='whether to use original graph')
    parser.add_argument("--simulated", action="store_true",
                        help='whether to use simulated graph')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.info(args)
    main(args)
