import os

import numpy as np
import logging
import tvm
from tvm import relay, autotvm
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import tvm.contrib.graph_runtime as runtime


def simple_mod(dshape, ishape, axis=0):
    d = relay.var("d", relay.TensorType(dshape, "float32"))
    i = relay.var("i", relay.TensorType(ishape, "int64"))
    u = relay.var("u", relay.TensorType(ishape, "float32"))
    z = relay.op.scatter(d, i, u, axis)
    func = relay.Function([d, i, u], z)
    mod = tvm.IRModule()
    mod["main"] = func
    return mod, {}


target = "cuda"

#### TUNING OPTION ####
network = "scatter"
log_file = "%s.log" % network

tuning_option = {
    "log_filename": log_file,
    "tuner": "xgb",
    "n_trial": 10,
    "early_stopping": 10,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
    ),
}

def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=1000,
    early_stopping=None,
    log_filename="tuning.log",
    use_transfer_learning=True,
):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)

def tune_and_evaluate(tuning_opt):
    # extract workloads from relay program
    print("Extract tasks...")
    size = (5000, )
    dshape = ishape = size
    axis = 0
    mod, params = simple_mod(size, size, axis)
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params, ops=(relay.op.get("scatter"),)
    )

    # run tuning tasks
    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)

    # compile kernels with history best records
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(mod, target=target, params=params)

        # load parameters
        ctx = tvm.context(str(target), 0)
        module = runtime.GraphModule(lib["default"](ctx))

        data_np = np.random.uniform(size=dshape).astype("float32")
        updates_np = np.random.uniform(size=ishape).astype("float32")
        indices_np = np.random.randint(-dshape[axis], dshape[axis] - 1, ishape).astype("int64")

        module.set_input("d", data_np)
        module.set_input("i", indices_np)
        module.set_input("u", updates_np)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=600)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print(
            "Mean inference time (std dev): %.2f ms (%.2f ms)"
            % (np.mean(prof_res), np.std(prof_res))
        )


# We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.
logging.basicConfig(level=logging.DEBUG)
tune_and_evaluate(tuning_option)
