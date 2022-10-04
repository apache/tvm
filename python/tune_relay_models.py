import os

import numpy as np
import pickle
# import torch.jit

import tvm
from tvm import relay, autotvm
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.measure.measure_methods import set_cuda_target_arch
import tvm.contrib.graph_executor as runtime

import asyncio

if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# python -m tvm.exec.query_rpc_tracker --host=127.0.0.1 --port=9190
# python -m tvm.exec.rpc_tracker --host=127.0.0.1 --port=9190
# python -m tvm.exec.rpc_tracker --host=192.168.6.252 --port=9190

# target = tvm.target.Target("opencl", host="llvm")
# target = "llvm -mtriple=i386-unknown-windows-msvc -mattr=+sse2"


target = tvm.target.Target("cuda -arch=sm_62", host="llvm -mtriple=aarch64-linux-gnu")
# set_cuda_target_arch('sm_62')
# os.environ["TVM_NVCC_PATH"] = "/usr/local/cuda-10.1/bin/nvcc"
# target = tvm.target.Target("cuda")
# model_path = "/home/share/data/workspace/project/python/yolov7/yolov7/runs/train/yolov7_waren" \
#              "/waren_20220730_2class_v7.torchscript.pt"


# model_path = "/home/share/data/workspace/project/nn_compiler/tvm/apps/waren20220712_4class_torch_v7tiny.torchscript
# .pt"
model_path = "D:\\workspace\\project\\nn_compiler\\tvm\\python\\Deploy"
model_name = "face_det"
target_name = "arm_cuda"
device_name = "tx2"
dtype = "float16"
tracker_host = "192.168.6.69"
# model_path = "D:/workspace/project/nn_compiler/tvm/apps/waren20220712_4class_torch_v7tiny.torchscript.pt"

log_file = "%s_%s_%s_%s_v10.log" % (model_name, target_name, device_name, dtype)

# tuning_option = {
#     "log_filename": log_file,
#     "tuner": "xgb",
#     "n_trial": 200,
#     # "n_parallel": 1,
#     "early_stopping": 600,
#     "measure_option": autotvm.measure_option(
#         builder=autotvm.LocalBuilder(timeout=40, n_parallel=1),
#         runner=autotvm.LocalRunner(number=10, repeat=3, timeout=40, min_repeat_ms=150),
#     ),
# }
# tuning_option = {
#     "log_filename": log_file,
#     "tuner": "xgb",
#     "n_trial": 20,
#     # "n_parallel": 1,
#     "early_stopping": 600,
#     "measure_option": autotvm.measure_option(
#         builder=autotvm.LocalBuilder(timeout=40, n_parallel=10),
#         runner=autotvm.RPCRunner("x86_64_252", host="192.168.6.252",
#                                  port=9190, number=1, repeat=3, timeout=40, min_repeat_ms=150),
#     ),
# }
# tuning_option = {
#     "log_filename": log_file,
#     "tuner": "xgb",
#     "n_trial": 20,
#     # "n_parallel": 1,
#     "early_stopping": 600,
#     "measure_option": autotvm.measure_option(
#         builder=autotvm.LocalBuilder(timeout=40, n_parallel=1),
#         runner=autotvm.RPCRunner("win32", host="127.0.0.1",
#                                  port=9190, number=1, repeat=3, timeout=40, min_repeat_ms=150),
#     ),
# }
tuning_option = {
    "log_filename": log_file,
    "tuner": "xgb",
    "n_trial": 1600,
    # "n_parallel": 1,
    "early_stopping": 600,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=4000, n_parallel=1, do_fork=False),
        runner=autotvm.RPCRunner(device_name, host=tracker_host,
                                 port=9190, number=1, repeat=3, timeout=10, min_repeat_ms=3),
    ),
    "use_transfer_learning": True,
}


# You can skip the implementation of this function for this tutorial.
def tune_tasks(
        tasks,
        measure_option,
        tuner="xgb",
        n_trial=1000,
        early_stopping=None,
        log_filename="tuning.log",
        use_transfer_learning=False,
):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    # tmp_log_file = "./tuning.log.tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        print(prefix)

        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
            print('finish create grid search tuner')
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.exists(tmp_log_file):
                autotvm.record.pick_best(tmp_log_file, log_filename)
                exit(0)
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        print('begin tak : ' + str(tsk_trial))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            # n_parallel=1,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.

def tune_and_evaluate(tuning_opt):
    # extract workloads from relay program
    print("Extract tasks...")
    # torchscript_model = torch.jit.load(model_path,
    #                                    map_location=torch.device('cpu'))
    input_name = "images"
    img = np.zeros((1, 3, 640, 640), dtype=np.float)

    mod_path = os.path.join(model_path, "%s_%s.pickle" % (model_name, dtype))
    params_path = os.path.join(model_path, "%s_%s.params" % (model_name, dtype))
    with open(mod_path, "rb") as mod_fn:
        mod_raw = mod_fn.read()
        mod = pickle.loads(mod_raw)
        print(mod)
    with open(params_path, "rb") as params_fn:
        params_raw = params_fn.read()
        params_array = bytearray(params_raw)
        params = relay.load_param_dict(params_array)
    # shape_dict = {input_name: img.shape}
    # shape_list = [(input_name, img.shape)]
    # input_shape = (1, 3, 640, 640)
    # output_shape = (1, 3, 80, 80, 7)
    #
    # mod, params = relay.frontend.from_pytorch(torchscript_model, shape_list)

    # mod, params, input_shape, out_shape = get_network(network, batch_size=1)
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params, ops=(relay.op.get("nn.conv2d"),)
    )
    ntasks = []
    if os.path.exists(log_file):
        with autotvm.apply_history_best(log_file) as appHistBest:
            for task in tasks:
                # print("query for task " + str(task))
                mes = appHistBest.query(task.target, task.workload)
                print("query result is "+str(mes))
                if str(mes) == ",None":
                    ntasks.append(task)
                else:
                    print("already tuned \n")
    else:
        ntasks = list(tasks)

    # run tuning tasks
    print("Tuning...%d/%d" % (len(ntasks), len(tasks)))
    tune_tasks(ntasks, **tuning_opt)

    # compile kernels with history best records
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(mod, target=target, params=params)

        path = "./lib_yolov7_tiny_cuda_tx2.so"
        lib.export_library(path)
        # load parameters
        remote = autotvm.measure.request_remote("tx2", "192.168.6.252",
                                                9190, timeout=10000)
        remote.upload(path)
        rlib = remote.load_module("lib_yolov7_tiny_cuda_tx2.so")

        dev = remote.device(str(target), 0)
        module = runtime.GraphModule(rlib["default"](dev))
        data_tvm = tvm.nd.array((np.random.uniform(size=img.shape)).astype(dtype), dev)
        module.set_input("images", data_tvm)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", dev, number=1, repeat=30)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print(
            "Mean inference time (std dev): %.2f ms (%.2f ms)"
            % (np.mean(prof_res), np.std(prof_res))
        )


if __name__ == '__main__':
    tune_and_evaluate(tuning_option)
