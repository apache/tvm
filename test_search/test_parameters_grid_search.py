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
    # Define the grid search parameters
    slide_window_sizes = [
        10,
        20,
        30,
        40,
        50,
        60,
    ]  # Example values for sliding window size
    n_starts = [
        4,
        8,
        12,
        16,
        20,
        24,
        28,
        32,
    ]  # Example values for number of start points
    max_tuning_time = 120  # Maximum tuning time in seconds
    max_trials = 1000  # Maximum number of measurement trials
    init_population_size = 64
    predict_score_threshold_ratio = 0.6  # Threshold for the predict score
    measure_threshold_ratio = 0.6  # Threshold for the measured throughput

    best_time = float("inf")
    best_params = None

    for slide_window_size in slide_window_sizes:
        for n_start in n_starts:
            start_time = time.time()
            with tempfile.TemporaryDirectory() as work_dir:
                tuner = ms.dynamic_gradient_search.DynamicGradientSearchTuner(
                    matmul,
                    n_start,
                    init_population_size,
                    slide_window_size,
                    max_trials,
                    target=Target("llvm -mcpu=icelake-server -num-cores 28"),
                    task_name="matmul",
                    tmpdir=work_dir,
                )

                # Run tuner
                database = tuner.dynamic_gradient_search()
                end_time = time.time()
                search_time = end_time - start_time
                search_time /= 60  # Convert to minutes
                print(
                    f"Slide Window Size: {slide_window_size}, n_start: {n_start}, Total search time: {search_time} minutes",
                    flush=True,
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
                buff_c = tvm.nd.array(
                    np.zeros((1024, 1024), dtype="float32"), dev
                )
                myfunc(buff_a, buff_b, buff_c)
                tvm.testing.assert_allclose(buff_c.numpy(), c_np, rtol=1e-3)
                print("[INFO]*************Success!")

                evaluator = myfunc.time_evaluator(
                    myfunc.entry_name, dev, repeat=1000, number=1
                )
                eval_time = (
                    evaluator(buff_a, buff_b, buff_c).mean * 1000
                )  # Convert to ms
                print(f"Evaluator time: {eval_time} ms")

                # Update best parameters if the current search time is better
                if eval_time < best_time:
                    best_time = eval_time
                    best_params = (slide_window_size, n_start)

            else:
                print("[INFO]*************Failed!")

    print(
        f"[INFO] Best parameters found: slide_window_size={best_params[0]}, n_start={best_params[1]} with time: {best_time} ms"
    )


if __name__ == "__main__":
    test_dynamic_gradient_descent()
