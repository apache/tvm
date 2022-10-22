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


def _get_qcheckpointbias_runtime(mod):
    func = mod["main"]
    func = _quantize.CreateQCheckPointBiasCollector(func)

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

def _get_qcheckpointzpi_runtime(mod):
    func = mod["main"]
    func = _quantize.CreateQCheckPointZpiCollector(func)

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

def _get_qcheckpointzpw_runtime(mod):
    func = mod["main"]
    func = _quantize.CreateQCheckPointZpwCollector(func)

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

def _get_qcheckpointweight_runtime(mod):
    func = mod["main"]
    func = _quantize.CreateQCheckPointWeightCollector(func)

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


def _get_qcheckpointsi_runtime(mod):
    func = mod["main"]
    func = _quantize.CreateQCheckPointSiCollector(func)

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

def _get_qcheckpointsw_runtime(mod):
    func = mod["main"]
    func = _quantize.CreateQCheckPointSwCollector(func)

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

def _get_qcheckpointinput_runtime(mod):
    func = mod["main"]
    func = _quantize.CreateQCheckPointInputCollector(func)

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

def _get_qcheckpointaddoutput_runtime(mod):
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


def collect_wj_3(mod_quantize, dataset=None):
    assert dataset
    # Path to save feature maps and weights
    print("Start Collecting_wj...")
    cfg = current_qconfig()
    root_dir_name = cfg.get_rootdir_name() + "/dbug_qfm/"

    QCheckpoint_bias_dir = root_dir_name + "Bias"
    QCheckpoint_zpi_dir = root_dir_name + "Zpi"
    QCheckpoint_zpw_dir = root_dir_name + "Zpw"
    QConWeight_dir = root_dir_name + "QConWeight"
    QConInput_dir = root_dir_name + "QConInput"
    QAddOutput_dir = root_dir_name + "QAddOutput"
    #QSi_dir = root_dir_name + "QSi"
    #QSw_dir = root_dir_name + "QSw"
    QSiSw_dir = root_dir_name + "QSiSw"


    if not os.path.exists(root_dir_name):
        os.mkdir(root_dir_name)
    if not os.path.exists(QCheckpoint_zpi_dir):
        os.mkdir(QCheckpoint_zpi_dir)
    if not os.path.exists(QConWeight_dir):
        os.mkdir(QConWeight_dir)
    if not os.path.exists(QConInput_dir):
        os.mkdir(QConInput_dir)
    if not os.path.exists(QCheckpoint_bias_dir):
        os.mkdir(QCheckpoint_bias_dir)
    if not os.path.exists(QCheckpoint_zpw_dir):
        os.mkdir(QCheckpoint_zpw_dir)    
    if not os.path.exists(QAddOutput_dir):
        os.mkdir(QAddOutput_dir)       
    if not os.path.exists(QSiSw_dir):
        os.mkdir(QSiSw_dir)          
        
    print("starting sisw")
    conv_runtime = _get_qcheckpointsi_runtime(mod_quantize)
    num_psum_outputs = conv_runtime.get_num_outputs()
    for i in range(1):
        batch = dataset[i]
        conv_runtime.set_input(**batch)
        conv_runtime.run()
        for j in range(2):
            psum_tmp = conv_runtime.get_output(j).numpy()
            np.save(QSiSw_dir + "/" + "SiSw_{}".format(i*(num_psum_outputs) + j), psum_tmp)
    print("sisw collect done")  
    

          
    print("starting bias")
    conv_runtime = _get_qcheckpointbias_runtime(mod_quantize)
    num_psum_outputs = conv_runtime.get_num_outputs()
    for i in range(1):
        batch = dataset[i]
        conv_runtime.set_input(**batch)
        conv_runtime.run()
        for j in range(2):
            psum_tmp = conv_runtime.get_output(j).numpy()
            np.save(QCheckpoint_bias_dir + "/" + "Bias_{}".format(i*(num_psum_outputs) + j), psum_tmp)
    print("bias collect done")    
    
    
    print("starting zpi")
    conv_runtime = _get_qcheckpointzpi_runtime(mod_quantize)
    num_psum_outputs = conv_runtime.get_num_outputs()
    for i in range(1):
        batch = dataset[i]
        conv_runtime.set_input(**batch)
        conv_runtime.run()
        for j in range(2):
            psum_tmp = conv_runtime.get_output(j).numpy()
            np.save(QCheckpoint_zpi_dir + "/" + "Zpi_{}".format(i*(num_psum_outputs) + j), psum_tmp)
    print("zpi collect done")    
    
    
    print("starting zpw")
    conv_runtime = _get_qcheckpointzpw_runtime(mod_quantize)
    num_psum_outputs = conv_runtime.get_num_outputs()
    for i in range(1):
        batch = dataset[i]
        conv_runtime.set_input(**batch)
        conv_runtime.run()
        for j in range(2):
            psum_tmp = conv_runtime.get_output(j).numpy()
            np.save(QCheckpoint_zpw_dir + "/" + "Zpw_{}".format(i*(num_psum_outputs) + j), psum_tmp)
    print("zpw collect done")    
    
    print("starting weight")
    conv_runtime = _get_qcheckpointweight_runtime(mod_quantize)
    num_psum_outputs = conv_runtime.get_num_outputs()
    for i in range(1):
        batch = dataset[i]
        conv_runtime.set_input(**batch)
        conv_runtime.run()
        for j in range(2):
            psum_tmp = conv_runtime.get_output(j).numpy()
            np.save(QConWeight_dir + "/" + "ConvWeight_{}".format(i*(num_psum_outputs) + j), psum_tmp)
    print("weight collect done")    
    
    print("starting input")
    conv_runtime = _get_qcheckpointinput_runtime(mod_quantize)
    num_psum_outputs = conv_runtime.get_num_outputs()
    for i in range(1):
        batch = dataset[i]
        conv_runtime.set_input(**batch)
        conv_runtime.run()
        for j in range(2):
            psum_tmp = conv_runtime.get_output(j).numpy()
            np.save(QConInput_dir + "/" + "ConvInput_{}".format(i*(num_psum_outputs) + j), psum_tmp)
    print("input collect done")    
    
    
    print("starting addout")
    conv_runtime = _get_qcheckpointaddoutput_runtime(mod_quantize)
    num_psum_outputs = conv_runtime.get_num_outputs()
    for i in range(1):
        batch = dataset[i]
        conv_runtime.set_input(**batch)
        conv_runtime.run()
        for j in range(2):
            psum_tmp = conv_runtime.get_output(j).numpy()
            np.save(QAddOutput_dir + "/" + "AddOutput_{}".format(i*(num_psum_outputs) + j), psum_tmp)
    print("addout collect done")   