import mxnet as mx
import tvm
from tvm import relay
from tvm import hago
from mxnet import gluon

import logging
logging.basicConfig(level=logging.DEBUG)
def get_calibration_dataset(dataset, batch_fn, var_name, num_samples=100):
    dataset.reset()
    batches = []
    for i, batch in enumerate(dataset):
        if i * dataset.batch_size > num_samples:
            break
        data, label = batch_fn(batch, [mx.cpu(0)])
        batches.append({var_name: tvm.nd.array(data[0].asnumpy()),
                        'label': tvm.nd.array(label[0].asnumpy())})
    return hago.CalibrationDataset(batches)


##################
# Evaluation infra
##################
def eval_acc(func, dataset, batch_fn, args, var_name, target='cuda', ctx=tvm.gpu(), postprocess=None, log_interval=100):
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(func, target)
    # create runtime module
    m = tvm.contrib.graph_runtime.create(graph, lib, ctx)
    m.set_input(**params)

    # setup evaluaiton metric
    dataset.reset()
    batch_size = dataset.batch_size
    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)
    acc_top1.reset()
    acc_top5.reset()
    # Execute

    if args.soundness_check:
        exit_at_batch = (100 + batch_size - 1)//batch_size
    else:
        exit_at_batch = -1

    for i, batch in enumerate(dataset):
        data, label = batch_fn(batch, [mx.cpu(0)])
        m.set_input(var_name, data[0].asnumpy())
        m.run()
        out_arr = m.get_output(0).asnumpy()
        if postprocess is not None:
            out_arr = postprocess(out_arr)
        acc_top1.update(label, [mx.nd.array(out_arr)])
        acc_top5.update(label, [mx.nd.array(out_arr)])

        if not (i + 1) % log_interval or i == exit_at_batch:
            _, top1 = acc_top1.get()
            _, top5 = acc_top5.get()
            nsamples = (i + 1) * batch_size
            logging.info('[%d samples] validation: acc-top1=%f acc-top5=%f', nsamples, top1, top5)

        if i == exit_at_batch:
            break
    logging.info('[final] validation: acc-top1=%f acc-top5=%f', top1, top5)
    return top1


#################
# Quantize helper
#################
def quantize_hago(mod, params, calib_dataset,
                  qconfig=None, hardware=None, tuner=None,
                  target="llvm", ctx=tvm.cpu(), eval_only=False):
    if qconfig is None:
        qconfig = hago.qconfig(log_file='temp.log')
    if hardware is None:
        hardware = hago.create_accelerator_description()

    with qconfig:
        graph = hago.prerequisite_optimize(mod['main'], params=params)
        logging.debug('current quantize config')
        logging.debug(hago.current_qconfig())
        space = hago.generate_search_space(graph, hardware)
        if tuner is None:
            tuner = hago.DefaultSetting(space, 'accuracy')
        elif isinstance(tuner, list):
            tuner = hago.DefaultSetting(space, 'accuracy', tuner)
        elif tuner == 'greedy':
            tuner = hago.GreedySearchTuner(space, "accuracy")
        elif tuner == 'batched':
            tuner = hago.BatchedGreedySearchTuner(space, "accuracy")

        if eval_only:
            record = hago.pick_best(qconfig.log_file, "accuracy")
            print(record)
            raise ValueError
        else:
            strategy, result = hago.search_quantize_strategy(graph, hardware, calib_dataset, tuner, ctx, target)
        print('strategy')
        print(strategy)

        quantizer = hago.create_quantizer(graph, hardware, strategy)
        simulated_graph = quantizer.simulate()
        quantized_graph = quantizer.quantize()
        lowered_quantized_graph = relay.qnn.transform.CanonicalizeOps()(tvm.IRModule.from_expr(quantized_graph))
        logging.debug('simulated graph')
        logging.debug(simulated_graph.astext(show_meta_data=False))
        logging.debug('quantize graph')
        logging.debug(quantized_graph.astext(show_meta_data=False))
        logging.debug('lowered quantized graph')
        logging.debug(lowered_quantized_graph.astext(show_meta_data=False))
        # hago.inspect_graph_statistic(graph, hardware, strategy, dataset, ctx, target='llvm')
        return tvm.IRModule.from_expr(quantized_graph)

def target_and_ctx(device):
    if device == 'cpu':
        target = 'llvm'
        ctx = tvm.cpu()
    elif device == 'gpu':
        target = 'cuda'
        ctx = tvm.gpu(1)
    return target, ctx
