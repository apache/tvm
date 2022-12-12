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



def collect_wj_7(mod_quantize, dataset=None):
    assert dataset
    # Path to save feature maps and weights
    print("Start Collecting_wj...")
    cfg = current_qconfig()
    root_dir_name = cfg.get_rootdir_name() + "/dbug_qfm/"

    QCheckpoint_matmul_dir = root_dir_name + "MatMul"

    if not os.path.exists(root_dir_name):
        os.mkdir(root_dir_name)

    if not os.path.exists(QCheckpoint_matmul_dir):
        os.mkdir(QCheckpoint_matmul_dir)


    print("starting Matmu;")
    addoutput_max = []
    addoutput_min = []
    addoutput_maxmin = []
    conv_runtime = _get_qcheckpoint_runtime(mod_quantize)
    num_psum_outputs = conv_runtime.get_num_outputs()
    if(num_psum_outputs != 0):
        for i in range(len(dataset)):
            batch = dataset[i]
            conv_runtime.set_input(**batch)
            conv_runtime.run()
            for j in range(num_psum_outputs):
                psum_tmp = conv_runtime.get_output(j).numpy()
                np.save(QCheckpoint_matmul_dir + "/" + "CheckPoint_{}".format(i*(num_psum_outputs) + j), psum_tmp)
                addoutput_max.append(np.max(psum_tmp))
                addoutput_min.append(np.min(psum_tmp))
        addoutput_maxmin.append(np.array(addoutput_max).max())
        addoutput_maxmin.append(np.array(addoutput_min).max())
        print("check_point collect done")    
    else:
        addoutput_maxmin = [0]
        
    

    return addoutput_maxmin
