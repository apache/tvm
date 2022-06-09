"""SSD Minimum Reproducible Example"""
from os import path as osp

import tvm
from tvm.meta_schedule import ApplyHistoryBest
from tvm.meta_schedule.database import JSONDatabase
from tvm.meta_schedule.utils import autotvm_silencer
from tvm.relay.frontend import from_onnx
from tvm.relay import vm
from tvm.ir.transform import PassContext
import numpy as np
import onnx


if __name__ == "__main__":
    work_dir = "/home/mbs/github/mbs-tvm/tests/python/relay/"
    onnx_model = onnx.load_model(osp.join(work_dir, "ssd.onnx"))
    database = JSONDatabase(
        osp.join(work_dir, "database_workload.json"),
        osp.join(work_dir, "database_tuning_record.json"),
    )
    target = tvm.target.Target("nvidia/geforce-rtx-3070")
    dev = tvm.cuda()
    shape_dict = {"image": [1, 3, 1200, 1200]}
    datas = {
        "image": tvm.nd.array(np.random.randn(1, 3, 1200, 1200).astype("float32"), dev),
    }
    mod, params = from_onnx(onnx_model, shape_dict, freeze_params=True)
    print("----------------------")
    print(mod)
    print("----------------------")
    with target, autotvm_silencer(), ApplyHistoryBest(database):
        with PassContext(
                opt_level=3,
                config={"relay.backend.use_meta_schedule": True},
        ):
            vm_exec = vm.compile(mod, target=target, params=params)
