from auto_tune_utils import *

if __name__ == '__main__':
    target = tvm.target.Target("cuda -arch=sm_62", host="llvm -mtriple=aarch64-linux-gnu")
    autoT = AutoTVMInst(target, "/home/share/data/workspace/project/nn_compiler/tvm/python/depoly",
                        "face_det", "arm_cuda", "float16", "192.168.6.252", "tx2"
                        )
    tuning_option = {
        "log_filename": autoT.log_file,
        "tuner": "xgb",
        "n_trial": 1600,
        # "n_parallel": 1,
        "early_stopping": 600,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=4000, n_parallel=1, do_fork=False),
            runner=autotvm.RPCRunner(autoT.device_name, host=autoT.tracker_host,
                                     port=9190, number=1, repeat=10, timeout=10, min_repeat_ms=3),
        ),
        "use_transfer_learning": False,
    }
    autoT.tune_and_evaluate(tuning_option)