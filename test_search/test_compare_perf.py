import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np

import tvm
import tvm.testing
from tvm import meta_schedule as ms
from tvm.meta_schedule.tune_context import _normalize_mod
from tvm.script import tir as T
from tvm.target import Target
from tvm.tir import Schedule


@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (1024, 1024), "float32")
    B = T.match_buffer(b, (1024, 1024), "float32")
    C = T.match_buffer(c, (1024, 1024), "float32")
    for i, j, k in T.grid(1024, 1024, 1024):
        with T.block("matmul"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0  # type: ignore
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def test_dynamic_gradient_descent(tuning_time):
    print(f"gradient_descent tuning time: {tuning_time}s")
    slide_window_size = 10
    max_trials = 1000000
    n_start = 5
    init_population_size = 2048
    # Initialize tuner
    # start_time = time.time()
    with tempfile.TemporaryDirectory() as work_dir:
        tuner = ms.dynamic_gradient_search.DynamicGradientSearchTuner(
            matmul,
            n_start,
            init_population_size,
            slide_window_size,
            max_trials,
            max_tuning_time=tuning_time,
            target=Target("llvm -mcpu=icelake-server -num-cores 28"),
            task_name="matmul",
            tmpdir=work_dir,
        )

        # Run tuner
        database = tuner.dynamic_gradient_search()
        record = database.query_tuning_record(
            _normalize_mod(matmul),
            Target("llvm -mcpu=icelake-server -num-cores 28"),
            workload_name="main",
        )

        if record is not None:
            sch = Schedule(record.workload.mod)
            record.trace.apply_to_schedule(sch, remove_postproc=False)
            myfunc = tvm.build(
                sch.mod,
                target=Target("llvm -mcpu=icelake-server -num-cores 28"),
                name="matmul",
            )
            dev = tvm.device("cpu", 0)
            a_np = np.random.uniform(size=(1024, 1024)).astype("float32")
            b_np = np.random.uniform(size=(1024, 1024)).astype("float32")
            c_np = a_np.dot(b_np)
            buff_a = tvm.nd.array(a_np, dev)
            buff_b = tvm.nd.array(b_np, dev)
            buff_c = tvm.nd.array(np.zeros((1024, 1024), dtype="float32"), dev)
            myfunc(buff_a, buff_b, buff_c)
            tvm.testing.assert_allclose(buff_c.numpy(), c_np, rtol=1e-3)
            evaluator = myfunc.time_evaluator(
                myfunc.entry_name, dev, repeat=1000, number=1
            )
            op_time = evaluator(buff_a, buff_b, buff_c).mean * 1000
        else:
            op_time = 0.0
        return op_time


def test_meta_schedule(tuning_time):
    print(f"meta scheudle tuning time: {tuning_time}s")
    time.time()
    with tempfile.TemporaryDirectory() as work_dir:
        target = Target("llvm -mcpu=icelake-server -num-cores 28")
        database = ms.tune_tir(
            mod=matmul,
            target=target,
            max_trials_global=1000000,
            tuning_time=tuning_time,
            num_trials_per_iter=64,
            work_dir=work_dir,
            runner=ms.runner.LocalRunner(
                evaluator_config=ms.runner.EvaluatorConfig(
                    number=1,
                    repeat=10,
                    min_repeat_ms=10,
                )
            ),
            cost_model=ms.cost_model.XGBModel(
                extractor=ms.feature_extractor.PerStoreFeature(),
                adaptive_training=True,
            ),
            strategy=ms.search_strategy.EvolutionarySearch(),
        )
    record = database.query_tuning_record(
        _normalize_mod(matmul),
        Target("llvm -mcpu=icelake-server -num-cores 28"),
        workload_name="main",
    )
    if record is not None:
        sch = Schedule(record.workload.mod)
        record.trace.apply_to_schedule(sch, remove_postproc=False)
        myfunc = tvm.build(
            sch.mod,
            target=Target("llvm -mcpu=icelake-server -num-cores 28"),
            name="matmul",
        )
        dev = tvm.device("cpu", 0)
        a_np = np.random.uniform(size=(1024, 1024)).astype("float32")
        b_np = np.random.uniform(size=(1024, 1024)).astype("float32")
        c_np = a_np.dot(b_np)
        buff_a = tvm.nd.array(a_np, dev)
        buff_b = tvm.nd.array(b_np, dev)
        buff_c = tvm.nd.array(np.zeros((1024, 1024), dtype="float32"), dev)
        myfunc(buff_a, buff_b, buff_c)
        tvm.testing.assert_allclose(buff_c.numpy(), c_np, rtol=1e-3)

        evaluator = myfunc.time_evaluator(
            myfunc.entry_name, dev, repeat=1000, number=1
        )
        op_time = evaluator(buff_a, buff_b, buff_c).mean * 1000
    else:
        op_time = 0.0
    return op_time


if __name__ == "__main__":

    tuning_times = [
        30,
        60,
        90,
        120,
        150,
        180,
        210,
        240,
        300,
        360,
        420,
        480,
        540,
        600,
    ]
    gd_time = [test_dynamic_gradient_descent(time) for time in tuning_times]
    ev_time = [test_meta_schedule(time) for time in tuning_times]
    print("[INOT]********gd_time: ", gd_time)
    print("[INOT]********ev_time: ", ev_time)
    # 绘制曲线
    plt.figure(figsize=(10, 6))
    plt.plot(tuning_times, gd_time, label="Dynamic Gradient Descent Time")
    plt.plot(tuning_times, ev_time, label="Meta Schedule Time")

    # 添加标题和标签
    plt.title("Comparison of GD Time and EV Time")
    plt.xlabel("Tuning Time (s)")
    plt.ylabel("Latency (ms)")
    plt.legend()

    # 显示图形
    plt.savefig("./GD_VS_EV_compare.png")
