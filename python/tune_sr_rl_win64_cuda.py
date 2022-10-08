from auto_tune_utils import *

if __name__ == '__main__':
    target = tvm.target.Target("cuda -arch=sm_86", host="llvm -mtriple=x86_64-unknown-windows-msvc -mcpu=skylake")
    autoT = AutoTVMInst(target, "/home/share/data/workspace/project/nn_compiler/tvm/python/depoly",
                        "sr_rlfn", "win64_cuda", "float32", "192.168.6.252", "rtx3070-win-cuda"
                        )
    tuning_option = {
        "log_filename": autoT.log_file,
        "tuner": "xgb",
        "n_trial": 1600,
        # "n_parallel": 1,
        "early_stopping": 600,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=4000, n_parallel=8, do_fork=True),
            runner=autotvm.RPCRunner(autoT.device_name, host=autoT.tracker_host,
                                     port=9190, number=1, repeat=10, timeout=10, min_repeat_ms=3),
        ),
        "use_transfer_learning": False,
    }
    autoT.tune_and_evaluate(tuning_option)

