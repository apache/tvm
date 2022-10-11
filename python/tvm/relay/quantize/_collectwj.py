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

def collect_wj(mod_quantize, dataset=None):
    assert dataset
    # Path to save feature maps and weights
    print("Start Collecting_wj...")
    cfg = current_qconfig()
    root_dir_name = cfg.get_rootdir_name() + "/dbug_qfm/"

    QCheckpoint_dir = root_dir_name + "QCheckpoint"
    QPsum_dir = root_dir_name + "QPsum"


    if not os.path.exists(root_dir_name):
        os.mkdir(root_dir_name)

    if not os.path.exists(QCheckpoint_dir):
        os.mkdir(QCheckpoint_dir)

    print("starting conv_psum")
    psum = []
    conv_runtime = _get_qconv_runtime(mod_quantize)
    num_psum_outputs = conv_runtime.get_num_outputs()
    for i in range(len(dataset)):
        batch = dataset[i]
        conv_runtime.set_input(**batch)
        conv_runtime.run()
        for j in range(num_psum_outputs):
            psum_tmp = conv_runtime.get_output(j).numpy()
            np.save(QPsum_dir + "/" + "Psum_{}".format(i*(num_psum_outputs) + j), psum_tmp)
            psum.append(np.max(psum_tmp))
    psum = np.array(psum).max()
    print("conv_psum collect done")


    print("starting check_point")
    check_point = []
    check_point_runtime = _get_qcheckpoint_runtime(mod_quantize)
    num_check_point_outputs = check_point_runtime.get_num_outputs()
    print(len(dataset))
    print(num_check_point_outputs)
    for i in range(len(dataset)):
        batch = dataset[i]
        check_point_runtime.set_input(**batch)
        check_point_runtime.run()
        for j in range(num_check_point_outputs):
            check_point_tmp = check_point_runtime.get_output(j).numpy()
            np.save(QCheckpoint_dir + "/" + "CheckPoint_{}".format(i*(num_check_point_outputs) + j), check_point_tmp)
            check_point.append(np.max(check_point_tmp))
    check_point = np.array(check_point).max()
    print("check_point collect done")
   
    return psum, check_point
