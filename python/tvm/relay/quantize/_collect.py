#Author: zzk
"""Collect quantized weight or activation data"""
import tvm.ir
import tvm
import os
from tvm.runtime import Object
import numpy as np
import tqdm

from .quantize import QAnnotateKind, current_qconfig
from .. import op as _op
from .. import expr as _expr
from .. import analysis as _analysis
from .. import build_module as _build_module
from ...contrib import graph_executor
from .. import transform as _transform
from . import _quantize
from tvm import relay

def _get_relu_runtime(mod):
    func = mod["main"]
    func = _quantize.CreateReluCollector(func)

    if tvm.target.Target.current():
        target = tvm.target.Target.current()
        dev = tvm.device(target.kind.name)
    else:
        target = "llvm"
        dev = tvm.device(target)

    with tvm.transform.PassContext(opt_level=3):
        lib = _build_module.build(func, target=target)
    runtime = graph_executor.GraphModule(lib["default"](dev))

    return runtime

def _get_qcheckpoint_runtime(mod):
    func = mod["main"]
    func = _quantize.CreateQCheckPointCollector(func)

    if tvm.target.Target.current():
        target = tvm.target.Target.current()
        dev = tvm.device(target.kind.name)
    else:
        target = "llvm"
        dev = tvm.device(target)

    with tvm.transform.PassContext(opt_level=3):
        lib = _build_module.build(func, target=target)
    runtime = graph_executor.GraphModule(lib["default"](dev))

    return runtime


def _get_qadd_runtime(mod):
    func = mod["main"]
    func = _quantize.CreateQAddCollector(func)

    if tvm.target.Target.current():
        target = tvm.target.Target.current()
        dev = tvm.device(target.kind.name)
    else:
        target = "llvm"
        dev = tvm.device(target)

    with tvm.transform.PassContext(opt_level=3):
        lib = _build_module.build(func, target=target)
    runtime = graph_executor.GraphModule(lib["default"](dev))

    return runtime

def _get_qconv_runtime(mod):
    func = mod["main"]
    func = _quantize.CreateQConvCollector(func)

    if tvm.target.Target.current():
        target = tvm.target.Target.current()
        dev = tvm.device(target.kind.name)
    else:
        target = "llvm"
        dev = tvm.device(target)

    with tvm.transform.PassContext(opt_level=3):
        lib = _build_module.build(func, target=target)
    runtime = graph_executor.GraphModule(lib["default"](dev))

    return runtime

def _get_stats_runtime(mod):
    func = mod["main"]
    func = _quantize.CreateStatsCollector(func)

    if tvm.target.Target.current():
        target = tvm.target.Target.current()
        dev = tvm.device(target.kind.name)
    else:
        target = "llvm"
        dev = tvm.device(target)

    with tvm.transform.PassContext(opt_level=3):
        lib = _build_module.build(func, target=target)
    runtime = graph_executor.GraphModule(lib["default"](dev))

    return runtime

def _get_qweight_runtime(mod):
    func = mod["main"]
    func = _quantize.CreateQWeightCollector(func)

    if tvm.target.Target.current():
        target = tvm.target.Target.current()
        dev = tvm.device(target.kind.name)
    else:
        target = "llvm"
        dev = tvm.device(target)

    with tvm.transform.PassContext(opt_level=3):
        lib = _build_module.build(func, target=target)
    runtime = graph_executor.GraphModule(lib["default"](dev))

    return runtime

def _get_qactivation_runtime(mod):
    func = mod["main"]
    func = _quantize.CreateQActCollector(func)

    if tvm.target.Target.current():
        target = tvm.target.Target.current()
        dev = tvm.device(target.kind.name)
    else:
        target = "llvm"
        dev = tvm.device(target)

    with tvm.transform.PassContext(opt_level=3):
        lib = _build_module.build(func, target=target)
    runtime = graph_executor.GraphModule(lib["default"](dev))

    return runtime

def _get_qactivationall_runtime(mod):
    func = mod["main"]
    func = _quantize.CreateQActAllCollector(func)

    if tvm.target.Target.current():
        target = tvm.target.Target.current()
        dev = tvm.device(target.kind.name)
    else:
        target = "llvm"
        dev = tvm.device(target)

    with tvm.transform.PassContext(opt_level=3):
        lib = _build_module.build(func, target=target)
    runtime = graph_executor.GraphModule(lib["default"](dev))

    return runtime

def _generate_qactivation_map(mod):
    quantize_op = _op.get("relay.op.annotation.simulated_quantize")
    annotate_cast_op = _op.get("annotation.cast_hint")
    stop_fusion_op = _op.get("annotation.stop_fusion")
    split_op = _op.get("split")
    skip_quantizeop_node = [quantize_op, annotate_cast_op, stop_fusion_op, split_op]
    qact_all_list = []
    qact_list = []
    qact_qparam_list = [] # [scale, zero_point] format
    qweight_qparam_list = [] # [scale, zero_point] format

    def visit_func(expr):
        """visitor function for traverse"""
        if isinstance(expr, _expr.Call) and expr.op not in skip_quantizeop_node:
            qact_all_list.append(expr)
        if isinstance(expr, _expr.Call) and expr.op == quantize_op:
            attrs = expr.attrs
            kind = attrs.kind
            assert kind == QAnnotateKind.WEIGHT or kind == QAnnotateKind.INPUT or kind == QAnnotateKind.BIAS
            if kind == QAnnotateKind.WEIGHT:
                scale = expr.args[1].data.numpy()
                zero_point = expr.args[4].data.numpy()
                qweight_qparam_list.append([scale, zero_point])
            if kind == QAnnotateKind.INPUT:
                scale = expr.args[1].data.numpy()
                zero_point = expr.args[4].data.numpy()
                qact_qparam_list.append([scale, zero_point])
                last_op = expr.args[0]
                if not isinstance(last_op, _expr.Call):
                    last_op = "input"
                else:
                    while(isinstance(last_op, _expr.TupleGetItem) or last_op.op in skip_quantizeop_node):
                        if isinstance(last_op, _expr.TupleGetItem):
                            last_op = qact_all_list[-1]
                        else:
                            last_op = last_op.args[0]
                qact_list.append(last_op)

    main_func = mod["main"]
    _analysis.post_order_visit(main_func, visit_func)

    return qact_all_list, qact_list, qact_qparam_list, qweight_qparam_list

def collect(mod,mod_quantize, dataset=None):
    assert dataset
    # Path to save feature maps and weights
    print("Start Collecting...")
    cfg = current_qconfig()
    root_dir_name = cfg.get_rootdir_name() + "/dbug_qfm/"
    QWeight_int_dir = root_dir_name + "QWeight_int"
    QWeight_float_dir = root_dir_name + "QWeight_float"
    QActivation_int_dir = root_dir_name + "QActivation_int"
    QActivation_float_dir = root_dir_name + "QActivation_float"
    QCheckpoint_dir = root_dir_name + "QCheckpoint"
    QAnnotateCheckpoint_dir = root_dir_name + "QAnnotateCheckpoint"
    QSimualtedquantize_dir = root_dir_name + "QSimualtedquantize"

    if not os.path.exists(root_dir_name):
        os.mkdir(root_dir_name)
    if not os.path.exists(QWeight_int_dir):
        os.mkdir(QWeight_int_dir)
    if not os.path.exists(QWeight_float_dir):
        os.mkdir(QWeight_float_dir)
    if not os.path.exists(QActivation_int_dir):
        os.mkdir(QActivation_int_dir)
    if not os.path.exists(QActivation_float_dir):
        os.mkdir(QActivation_float_dir)
    if not os.path.exists(QCheckpoint_dir):
        os.mkdir(QCheckpoint_dir)
    if not os.path.exists(QAnnotateCheckpoint_dir):
        os.mkdir(QAnnotateCheckpoint_dir)
    if not os.path.exists(QSimualtedquantize_dir):
        os.mkdir(QSimualtedquantize_dir)

    """Step 1. get qactivation list and quantize params"""
    qact_all_list, qact_list, qact_qparam_list, qweight_qparam_list = \
            _generate_qactivation_map(mod)
    """
    #Step 2. get weight params (no bias)
    weight_runtime = _get_qweight_runtime(mod)
    num_weight_outputs = weight_runtime.get_num_outputs()
    assert(num_weight_outputs == len(qweight_qparam_list))
    batch = dataset[0]
    weight_runtime.set_input(**batch)
    weight_runtime.run()
    for j in range(num_weight_outputs):
        weight_float_tmp = weight_runtime.get_output(j).numpy()
        np.save(QWeight_float_dir + "/" + "QWeight_float_{}".format(j), weight_float_tmp)
        weight_int_tmp = np.round(np.add(np.divide(weight_float_tmp, qweight_qparam_list[j][0]), qweight_qparam_list[j][1]))
        np.save(QWeight_int_dir + "/" + "QWeight_int_{}".format(j), weight_int_tmp)

    #Step 3. get all act params
    actall_runtime = _get_qactivationall_runtime(mod)
    num_actall_outputs = actall_runtime.get_num_outputs()
    assert(num_actall_outputs == len(qact_all_list))
    batch_count = 0
    for batch in dataset:
        actall_runtime.set_input(**batch)
        actall_runtime.run()
        for j in range(num_actall_outputs):
            actall_float_tmp = actall_runtime.get_output(j).numpy()
            if not os.path.exists(QActivation_float_dir + "/" + str(batch_count)):
                os.mkdir(QActivation_float_dir + "/" + str(batch_count))
            np.save(QActivation_float_dir + "/" + str(batch_count) + "/" + qact_all_list[j].op.name + "_" + str(j), actall_float_tmp)
        batch_count += 1
    

    #Step 4. get q act params
    act_runtime = _get_qactivation_runtime(mod)
    num_act_outputs = act_runtime.get_num_outputs()
    assert(num_act_outputs == len(qact_qparam_list))
    batch_count = 0
    for batch in tqdm.tqdm(dataset):
        .set_input(**batch)
        act_runtime.run()
        for j in range(num_act_outputs):
            act_float_tmp = act_runtime.get_output(j).numpy()
            if not os.path.exists(QActivation_float_dir + "/" + str(batch_count)):
                os.mkdir(QActivation_float_dir + "/" + str(batch_count))
            if not os.path.exists(QActivation_int_dir + "/" + str(batch_count)):
                os.mkdir(QActivation_int_dir + "/" + str(batch_count))
            if qact_list[j] != "input":
                assert(os.path.exists(QActivation_float_dir + "/" + str(batch_count) + "/" + qact_list[j].op.name + "_" + str(qact_all_list.index(qact_list[j])) + ".npy"))
            np.save(QActivation_float_dir + "/" + str(batch_count) + "/" + (qact_list[j].op.name if qact_list[j] != "input" else "input") + "_" + (str(qact_all_list.index(qact_list[j])) if qact_list[j] != "input" else "input"), act_float_tmp)
            act_int_tmp = np.round(np.add(np.divide(act_float_tmp, qact_qparam_list[j][0]), qact_qparam_list[j][1]))
            np.save(QActivation_int_dir + "/" + str(batch_count) + "/" + (qact_list[j].op.name if qact_list[j] != "input" else "input") + "_" + (str(qact_all_list.index(qact_list[j])) if qact_list[j] != "input" else "input"), act_int_tmp)
        batch_count += 1
    """
    #Step 5. get all checkpoint int_output after realize
    print("starting checkpoint")
    checkpoint_runtime = _get_qcheckpoint_runtime(mod_quantize)
    num_checkpoint_outputs = checkpoint_runtime.get_num_outputs()
    batch = dataset[7]
    checkpoint_runtime.set_input(**batch)
    checkpoint_runtime.run()
    print("starting checkpoint done")
    for j in range(num_checkpoint_outputs):
        checkpoint_tmp = checkpoint_runtime.get_output(j).numpy()
        np.save(QCheckpoint_dir + "/" + "QCheckpoint_{}".format(j), checkpoint_tmp)
    
    #Step 6.
    simulatedquantize_runtime = _get_qactivation_runtime(mod)
    num_simualtedquantize_outputs = simulatedquantize_runtime.get_num_outputs()
    #assert(num_checkpoint_outputs == len(qact_all_list)+len(qweight_qparam_list))    
    batch = dataset[7]
    simulatedquantize_runtime.set_input(**batch)
    simulatedquantize_runtime.run()
    for j in range(num_simualtedquantize_outputs):
        simulatedquantize_tmp = simulatedquantize_runtime.get_output(j).numpy()
        simulatedquantize_int_tmp = np.round(np.add(np.divide(simulatedquantize_tmp, qact_qparam_list[j][0]), qact_qparam_list[j][1]))
        np.save(QSimualtedquantize_dir + "/" + "QSimulateQuantize_{}".format(j), simulatedquantize_int_tmp)
    """
    print("starting conv ")
    conv_runtime = _get_qcheckpoint_runtime(mod)
    num_add_outputs = conv_runtime.get_num_outputs()
    batch = dataset[7]
    conv_runtime.set_input(**batch)
    conv_runtime.run()
    for j in range(num_add_outputs):
        checkpoint_tmp = conv_runtime.get_output(j).numpy()
        np.save(QAnnotateCheckpoint_dir + "/" + "QCheckpoint_{}".format(j), checkpoint_tmp)
    """





    print("collect done")




    
    cosine_result_list = []

    dir = cfg.get_rootdir_name() + '/dbug_qfm/'
    for i in range(num_checkpoint_outputs):
        m1 = np.load(dir+'QCheckpoint/'+'QCheckpoint_'+str(i)+'.npy')
        m2 = np.load(dir+'QSimualtedquantize/'+'QCheckpoint_'+str(i)+'.npy')
        a= relay.quantize.get_consine_similar(m1,m2)
        cosine_result_list.append(a)

    print(cosine_result_list)
    f = open(root_dir_name+'cosine_result_list.txt','w')
    f.write(str(cosine_result_list))
    f.close()  