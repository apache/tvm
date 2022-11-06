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


def collect_wj_4(mod_quantize, dataset=None):
    assert dataset
    # Path to save feature maps and weights
    print("Start Collecting_wj...")
    cfg = current_qconfig()
    root_dir_name = cfg.get_rootdir_name() + "/dbug_qfm/"
    QAddOutput_dir = root_dir_name + "QAddPsum"

    if not os.path.exists(QAddOutput_dir):
        os.mkdir(QAddOutput_dir)       
        
    print("starting check_point")
    check_point_biass_max = []
    check_point_biass_min = []
    check_point_biass_maxmin = []
    check_point_runtime = _get_qcheckpointaddoutput_runtime(mod_quantize)
    num_check_point_outputs = check_point_runtime.get_num_outputs()
    print(len(dataset))
    print(num_check_point_outputs)
    for i in range(len(dataset)):
        batch = dataset[i]
        check_point_runtime.set_input(**batch)
        check_point_runtime.run()
        for j in range(num_check_point_outputs):
            check_point_tmp = check_point_runtime.get_output(j).numpy()
            np.save(QAddOutput_dir + "/" + "addpsum{}".format(i*(num_check_point_outputs) + j), check_point_tmp)
            check_point_biass_max.append(np.max(check_point_tmp))
            check_point_biass_min.append(np.min(check_point_tmp))
    check_point_biass_maxmin.append(np.array(check_point_biass_max).max())
    check_point_biass_maxmin.append(np.array(check_point_biass_min).min())

    print("check_point collect done")
    
    



    return check_point_biass_maxmin
