import tempfile
import time

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


def test_dynamic_gradient_descent():
    slide_window_size = 10  # Size of the sliding window used in dynamic gradient search
    max_tuning_time = 120  # Maximum tuning time in seconds, 120 is the suggested value
    max_trials = 1000000  # Maximum number of measurement trials to perform in dynamic gradient search, use 1000 to get better performance
    n_start = 5  # Number of start points from the initial sampled population
    # Number of samples to generate the initial model, 64 is the suggested
    # value
    init_population_size = 1024
    predict_score_threshold_ratio = 0.95  # Threshold for the predict score
    measure_threshold_ratio = 0.95  # Threshold for the measured throughput
    # Initialize tuner
    start_time = time.time()
    with tempfile.TemporaryDirectory() as work_dir:
        tuner = ms.dynamic_gradient_search.DynamicGradientSearchTuner(
            matmul,
            n_start,
            init_population_size,
            slide_window_size,
            max_trials,
            max_tuning_time=max_tuning_time,
            target=Target("llvm -mcpu=icelake-server -num-cores 28"),
            task_name="matmul",
            tmpdir=work_dir,
        )

        # Run tuner
        database = tuner.dynamic_gradient_search()
        end_time = time.time()
        search_time = end_time - start_time
        search_time /= 60
        print(f"Total search time: {search_time} minutes", flush=True)

        record = database.query_tuning_record(
            _normalize_mod(matmul),
            Target("llvm -mcpu=icelake-server -num-cores 28"),
            workload_name="main",
        )

    if record is not None:
        sch = Schedule(record.workload.mod)
        record.trace.apply_to_schedule(sch, remove_postproc=False)
        print("[INFO]final module: ", sch.mod)
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
        print("[INFO]*************Success!")
        evaluator = myfunc.time_evaluator(myfunc.entry_name, dev, repeat=1000, number=1)
        print(f"Evaluator time:  {evaluator(buff_a, buff_b, buff_c).mean * 1000} ms")
    else:
        print("[INFO]*************Failed!")


def test_meta_schedule():
    t1 = time.time()
    with tempfile.TemporaryDirectory() as work_dir:
        target = Target("llvm -mcpu=icelake-server -num-cores 28")
        database = ms.tune_tir(
            mod=matmul,
            target=target,
            max_trials_global=1000,
            tuning_time=120,
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
    t2 = time.time()
    search_time = t2 - t1
    search_time /= 60
    print(f"Meta schedule total search time: {search_time} minutes", flush=True)
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
        print("[INFO]*************Success!")
        evaluator = myfunc.time_evaluator(myfunc.entry_name, dev, repeat=1000, number=1)
        print(f"meta schedule Evaluator time:  {evaluator(buff_a, buff_b, buff_c).mean * 1000} ms")
    else:
        print("[INFO]*************Failed!")


if __name__ == "__main__":
    test_dynamic_gradient_descent()
    # test_meta_schedule()
