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


def _get_addoutput_runtime(mod):
    func = mod["main"]
    func = _quantize.CreateQCheckPointAddOutputCollector(func)

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

def _get_addpsum_runtime(mod):
    func = mod["main"]
    func = _quantize.CreateAddPsummCollector(func)

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

def _get_convadd_runtime(mod):
    func = mod["main"]
    func = _quantize.CreateConvAddCollector(func)

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

    
def _get_convpsum_runtime(mod):
    func = mod["main"]
    func = _quantize.CreateConvPsumCollector(func)

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


def collect_wj_5(mod_quantize, dataset=None):
    assert dataset
    # Path to save feature maps and weights
    print("Start Collecting_wj...")
    cfg = current_qconfig()
    root_dir_name = cfg.get_rootdir_name() + "/dbug_qfm/"

    QCheckpoint_addscale_dir = root_dir_name + "AddScale"
    QCheckpoint_convaddscale_dir = root_dir_name + "ConvAddScale"
    QCheckpoint_convadd_dir = root_dir_name + "ConvAdd"
    QCheckpoint_addoutput_dir = root_dir_name + "Addoutput"



    if not os.path.exists(root_dir_name):
        os.mkdir(root_dir_name)

    if not os.path.exists(QCheckpoint_addscale_dir):
        os.mkdir(QCheckpoint_addscale_dir)
        
    if not os.path.exists(QCheckpoint_convaddscale_dir):
        os.mkdir(QCheckpoint_convaddscale_dir)
        
    if not os.path.exists(QCheckpoint_convadd_dir):
        os.mkdir(QCheckpoint_convadd_dir)
    
    if not os.path.exists(QCheckpoint_addoutput_dir):
        os.mkdir(QCheckpoint_addoutput_dir)


    print("starting Addoutput")
    addoutput_max = []
    addoutput_min = []
    addoutput_maxmin = []
    conv_runtime = _get_addoutput_runtime(mod_quantize)
    num_psum_outputs = conv_runtime.get_num_outputs()
    if(num_psum_outputs != 0):
        for i in range(len(dataset)):
            batch = dataset[i]
            conv_runtime.set_input(**batch)
            conv_runtime.run()
            for j in range(num_psum_outputs):
                psum_tmp = conv_runtime.get_output(j).numpy()
                np.save(QCheckpoint_addscale_dir + "/" + "CheckPoint_{}".format(i*(num_psum_outputs) + j), psum_tmp)
                addoutput_max.append(np.max(psum_tmp))
                addoutput_min.append(np.min(psum_tmp))
        addoutput_maxmin.append(np.array(addoutput_max).max())
        addoutput_maxmin.append(np.array(addoutput_min).max())
        print("check_point collect done")    
    else:
        addoutput_maxmin = [0]
        
    print("starting AddScale")
    psum_max = []
    psum_min = []
    psum_maxmin = []
    conv_runtime = _get_addpsum_runtime(mod_quantize)
    num_psum_outputs = conv_runtime.get_num_outputs()
    if(num_psum_outputs != 0):
        for i in range(len(dataset)):
            batch = dataset[i]
            conv_runtime.set_input(**batch)
            conv_runtime.run()
            for j in range(num_psum_outputs):
                psum_tmp = conv_runtime.get_output(j).numpy()
                np.save(QCheckpoint_addscale_dir + "/" + "CheckPoint_{}".format(i*(num_psum_outputs) + j), psum_tmp)
                psum_max.append(np.max(psum_tmp))
                psum_min.append(np.min(psum_tmp))
        psum_maxmin.append(np.array(psum_max).max())
        psum_maxmin.append(np.array(psum_min).max())
        print("check_point collect done")    
    else:
        psum_maxmin = [0]
        
        
    print("starting ConvAddScale")
    check_point_siso_max = []
    check_point_siso_min = []
    check_point_siso_maxmin = []
    check_point_runtime = _get_convpsum_runtime(mod_quantize)
    num_check_point_outputs = check_point_runtime.get_num_outputs()
    print(len(dataset))
    print(num_check_point_outputs)
    for i in range(len(dataset)):
        batch = dataset[i]
        check_point_runtime.set_input(**batch)
        check_point_runtime.run()
        for j in range(num_check_point_outputs):
            check_point_tmp = check_point_runtime.get_output(j).numpy()
            np.save(QCheckpoint_convaddscale_dir + "/" + "CheckPoint_{}".format(i*(num_check_point_outputs) + j), check_point_tmp)
            check_point_siso_max.append(np.max(check_point_tmp))
            check_point_siso_min.append(np.min(check_point_tmp))
            #check_point_siso_max.append(np.max(check_point_runtime.get_output(j).numpy()))
            #check_point_siso_min.append(np.min(check_point_runtime.get_output(j).numpy()))
    check_point_siso_maxmin.append(np.array(check_point_siso_max).max())
    check_point_siso_maxmin.append(np.array(check_point_siso_min).min())

    print("check_point collect done")
    
    print("starting convadd")
    check_point_biass_max = []
    check_point_biass_min = []
    check_point_biass_maxmin = []
    check_point_runtime = _get_convadd_runtime(mod_quantize)
    num_check_point_outputs = check_point_runtime.get_num_outputs()
    print(len(dataset))
    print(num_check_point_outputs)
    for i in range(len(dataset)):
        batch = dataset[i]
        check_point_runtime.set_input(**batch)
        check_point_runtime.run()
        for j in range(num_check_point_outputs):
            check_point_tmp = check_point_runtime.get_output(j).numpy()
            np.save(QCheckpoint_convadd_dir + "/" + "CheckPoint_{}".format(i*(num_check_point_outputs) + j), check_point_tmp)
            check_point_biass_max.append(np.max(check_point_tmp))
            check_point_biass_min.append(np.min(check_point_tmp))
            #check_point_biass_max.append(np.max(check_point_runtime.get_output(j).numpy()))
            #check_point_biass_min.append(np.min(check_point_runtime.get_output(j).numpy()))
    check_point_biass_maxmin.append(np.array(check_point_biass_max).max())
    check_point_biass_maxmin.append(np.array(check_point_biass_min).min())

    print("check_point collect done")
    

    return addoutput_maxmin, psum_maxmin, check_point_siso_maxmin, check_point_biass_maxmin
