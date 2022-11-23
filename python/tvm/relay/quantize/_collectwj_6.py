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

def _get_qcheckpointbiass_runtime(mod):
    func = mod["main"]
    func = _quantize.CreateQCheckPointBiasSCollector(func)

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

def collect_wj_6(mod_quantize, dataset=None):
    assert dataset
    # Path to save feature maps and weights
    print("Start Collecting_wj...")
    cfg = current_qconfig()
    root_dir_name = cfg.get_rootdir_name() + "/dbug_qfm/"
    QPsum_dir = root_dir_name + "QPsum"
    Qconcate_dir = root_dir_name + "Concate"


    if not os.path.exists(root_dir_name):
        os.mkdir(root_dir_name)

    if not os.path.exists(QPsum_dir):
        os.mkdir(QPsum_dir)
    if not os.path.exists(Qconcate_dir):
        os.mkdir(Qconcate_dir)
        
   
    print("starting si")
    
    psum_max = []
    psum_min = []
    psum_maxmin = []
    conv_runtime = _get_convadd_runtime(mod_quantize)
    num_psum_outputs = conv_runtime.get_num_outputs()
    for i in range(len(dataset[0:1])):
        batch = dataset[i]
        conv_runtime.set_input(**batch)
        conv_runtime.run()
        for j in range(num_psum_outputs):
            psum_tmp = conv_runtime.get_output(j).numpy()
            np.save(Qconcate_dir + "/" + "Psum_{}".format(i*(num_psum_outputs) + j), psum_tmp)
            psum_max.append(np.max(psum_tmp))
            psum_min.append(np.min(psum_tmp))
    # print(psum_max)
    # print(max(psum_max))
    # print(psum_max.index(max(psum_max)))
    
    # print(psum_min)
    # print(max(psum_min))
    # print(psum_min.index(max(psum_min)))
    
    psum_maxmin.append(np.array(psum_max).max())
    psum_maxmin.append(np.array(psum_min).max())
    print("si collect done")         

 
    return psum_maxmin
