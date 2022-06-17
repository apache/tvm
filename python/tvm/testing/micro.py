from functools import lru_cache
import json
from pathlib import Path
from contextlib import ExitStack
import tempfile

import tvm
from tvm.micro import get_microtvm_template_projects
from tvm.relay.backend import Executor, Runtime
from tvm.runtime.executor.aot_executor import AotModule
import tvm.testing.micro

@lru_cache
def get_supported_boards(platform: str):
    boards_path = Path(get_microtvm_template_projects(platform)) / "boards.json"
    with open(boards_path) as f:
        return json.load(f)


def get_target(platform: str, board: str):
    model = tvm.testing.micro.get_supported_boards(platform)[board]["model"]
    return str(tvm.target.target.micro(model))


def tune_model(platform, board, mod, params, tasks, num_trials, tuner_cls=tvm.autotvm.tuner.GATuner):
    """Tune a Relay module of a full model and return best result for each task"""
    assert len(tasks) > 0

    module_loader = tvm.micro.AutoTvmModuleLoader(
        template_project_dir=get_microtvm_template_projects(platform),
        project_options={
            f"{platform}_board": board,
            "project_type": "host_driven",
        },
    )

    builder = tvm.autotvm.LocalBuilder(
        n_parallel=1,
        build_kwargs={"build_option": {"tir.disable_vectorize": True}},
        do_fork=False,
        build_func=tvm.micro.autotvm_build_func,
        runtime=tvm.relay.backend.Runtime("crt", {"system-lib": True}),
    )
    runner = tvm.autotvm.LocalRunner(number=1, repeat=1, timeout=100, module_loader=module_loader)
    measure_option = tvm.autotvm.measure_option(builder=builder, runner=runner)

    results = StringIO()
    for task in tasks:
        tuner = tuner_cls(task)

        tuner.tune(
            n_trial=num_trials,
            measure_option=measure_option,
            callbacks=[
                tvm.autotvm.callback.log_to_file(results),
                tvm.autotvm.callback.progress_bar(num_trials, si_prefix="M"),
            ],
            si_prefix="M",
        )
        assert tuner.best_flops > 1

    return results


def create_aot_session(platform, board, target, mod, params, tune_logs=None, use_cmsis_nn=False):
    executor = Executor("aot")
    crt_runtime = Runtime("crt", {"system-lib": True})

    with ExitStack() as stack:
        config = {"tir.disable_vectorize": True}
        if use_cmsis_nn:
            config['relay.ext.cmsisnn.options'] = {'mcpu': target.mcpu}
        stack.enter_context(tvm.transform.PassContext(opt_level=3, config=config))
        if tune_logs is not None:
            stack.enter_context(tvm.autotvm.apply_history_best(tune_logs))

        lowered = tvm.relay.build(
            mod,
            target=target,
            params=params,
            runtime=crt_runtime,
            executor=executor,
        )
    parameter_size = len(tvm.runtime.save_param_dict(lowered.get_params()))
    print(f"Model parameter size: {parameter_size}")

    # Once the project has been uploaded, we don't need to keep it
    with tempfile.TemporaryDirectory() as folder:
        project = tvm.micro.generate_project(
            str(get_microtvm_template_projects(platform)),
            mod,
            folder,
            {
                f"{platform}_board": board,
                "project_type": "host_driven",
            },
        )
        project.build()
        project.flash()

    return tvm.micro.Session(project.transport())


# This utility functions was designed ONLY for one input / one output models
# where the outputs are confidences for different classes.
def evaluate_model_accuracy(aot_executor, input_data, true_labels):
    assert aot_executor.get_num_inputs() == 1
    assert aot_executor.get_num_outputs() == 1
    assert len(input_data) == len(true_labels)

    predicted_labels = []
    aot_runtimes = []
    for sample in input_data:
        aot_executor.get_input(0).copyfrom(sample)
        runtime = aot_executor.module.time_evaluator("run", 1, 1, 1)
        output = aot_executor.get_output(0).numpy()
        predicted_labels.append(output.argmax())
        aot_runtimes.append(runtime)

    num_correct = sum(u == v for u, v in zip(true_labels, predicted_labels))
    return num_correct / len(input_data), sum(aot_runtimes) / len(input_data)
