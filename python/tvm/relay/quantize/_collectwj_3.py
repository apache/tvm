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

def _get_qcheckpointsiso_runtime(mod):
    func = mod["main"]
    func = _quantize.CreateQCheckPointSiSoCollector(func)

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

    QCheckpoint_dir = root_dir_name + "QCheckpoint"
    QCheckpoint_siso_dir = root_dir_name + "QSiSo"
    QCheckpoint_biass_dir = root_dir_name + "BiasS"
    QCheckpoint_zpi_dir = root_dir_name + "Zpi"
    QPsum_dir = root_dir_name + "QPsum"
    QConWeight_dir = root_dir_name + "QConWeight"
    QConInput_dir = root_dir_name + "QConInput"
    QConOutput_dir = root_dir_name + "QConOutput"


    if not os.path.exists(root_dir_name):
        os.mkdir(root_dir_name)

    if not os.path.exists(QCheckpoint_dir):
        os.mkdir(QCheckpoint_dir)
        
    if not os.path.exists(QCheckpoint_siso_dir):
        os.mkdir(QCheckpoint_siso_dir)
        
    if not os.path.exists(QCheckpoint_biass_dir):
        os.mkdir(QCheckpoint_biass_dir)
    
    if not os.path.exists(QCheckpoint_zpi_dir):
        os.mkdir(QCheckpoint_zpi_dir)
    
    if not os.path.exists(QPsum_dir):
        os.mkdir(QPsum_dir)
    
    if not os.path.exists(QConWeight_dir):
        os.mkdir(QConWeight_dir)
        
    if not os.path.exists(QConInput_dir):
        os.mkdir(QConInput_dir)
    
    if not os.path.exists(QConOutput_dir):
        os.mkdir(QConOutput_dir)
        
    print("starting conv_psum")
    conv_runtime = _get_qconv_runtime(mod_quantize)
    num_psum_outputs = conv_runtime.get_num_outputs()
    for i in range(1):
        batch = dataset[i]
        conv_runtime.set_input(**batch)
        conv_runtime.run()
        for j in range(2):
            psum_tmp = conv_runtime.get_output(j).numpy()
            np.save(QConOutput_dir + "/" + "Psum_{}".format(i*(num_psum_outputs) + j), psum_tmp)
    print("conv_psum collect done")    
    
    
    print("starting conv_input")
    conv_runtime = _get_qcheckpoint_runtime(mod_quantize)
    num_psum_outputs = conv_runtime.get_num_outputs()
    for i in range(1):
        batch = dataset[i]
        conv_runtime.set_input(**batch)
        conv_runtime.run()
        for j in range(2):
            psum_tmp = conv_runtime.get_output(j).numpy()
            np.save(QConInput_dir + "/" + "Input_{}".format(i*(num_psum_outputs) + j), psum_tmp)
    print("conv_input collect done")    
    
    
    print("starting conv_output")
    conv_runtime = _get_qcheckpointsiso_runtime(mod_quantize)
    num_psum_outputs = conv_runtime.get_num_outputs()
    for i in range(1):
        batch = dataset[i]
        conv_runtime.set_input(**batch)
        conv_runtime.run()
        for j in range(2):
            psum_tmp = conv_runtime.get_output(j).numpy()
            np.save(QConWeight_dir + "/" + "Weight_{}".format(i*(num_psum_outputs) + j), psum_tmp)
    print("conv_output collect done")    
        
    

    
    
