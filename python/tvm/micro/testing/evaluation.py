# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
Provides high-level functions for instantiating and timing AOT models. Used
by autotuning tests in tests/micro, and may be used for more performance
tests in the future.

"""

import logging
from io import StringIO
from pathlib import Path
from contextlib import ExitStack
import tempfile
import shutil

import tvm
from tvm.relay.op.contrib import cmsisnn


def tune_model(
    platform,
    board,
    target,
    mod,
    params,
    num_trials,
    tuner_cls=tvm.autotvm.tuner.GATuner,
    project_options=None,
):
    """Autotunes a model with microTVM and returns a StringIO with the tuning logs"""
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        tasks = tvm.autotvm.task.extract_from_program(mod["main"], {}, target)
    assert len(tasks) > 0
    assert isinstance(params, dict)

    project_options = {
        "board": board,
        "project_type": "host_driven",
        **(project_options or {}),
    }

    module_loader = tvm.micro.AutoTvmModuleLoader(
        template_project_dir=tvm.micro.get_microtvm_template_projects(platform),
        project_options=project_options,
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
        # Note that we might not find a working schedule at all, in which case
        # tuner.best_flops would equal zero. This is not good, but checking for
        # this case will happen elsewhere.

    return results


def create_aot_session(
    platform,
    board,
    target,
    mod,
    params,
    build_dir=Path(tempfile.mkdtemp()),
    tune_logs=None,
    timeout_override=None,
    use_cmsis_nn=False,
    project_options=None,
    use_existing=False,
):
    """AOT-compiles and uploads a model to a microcontroller, and returns the RPC session"""

    executor = tvm.relay.backend.Executor("aot")
    crt_runtime = tvm.relay.backend.Runtime("crt", {"system-lib": True})

    with ExitStack() as stack:
        config = {"tir.disable_vectorize": True}
        if use_cmsis_nn:
            config["relay.ext.cmsisnn.options"] = {"mcpu": target.mcpu}
        stack.enter_context(tvm.transform.PassContext(opt_level=3, config=config))
        if use_cmsis_nn:
            mod = cmsisnn.partition_for_cmsisnn(mod, params, mcpu=target.mcpu)
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

    project_options = {
        "board": board,
        "project_type": "host_driven",
        # {} shouldn't be the default value for project options ({}
        # is mutable), so we use this workaround
        **(project_options or {}),
    }

    if use_existing:
        shutil.rmtree(build_dir / "project" / "build")
        project = tvm.micro.GeneratedProject.from_directory(
            build_dir / "project",
            options=project_options,
        )

    else:
        project = tvm.micro.generate_project(
            str(tvm.micro.get_microtvm_template_projects(platform)),
            lowered,
            build_dir / "project",
            project_options,
        )

    project.build()
    project.flash()
    return tvm.micro.Session(project.transport(), timeout_override=timeout_override)


def predict_labels_aot(session, aot_executor, input_data, runs_per_sample=1):
    """Predicts labels for each sample in input_data using host-driven AOT.
    Returns an iterator of (label, runtime) tuples. This function can only
    be used with models for which the output is the confidence for each class."""

    assert aot_executor.get_num_inputs() == 1
    assert aot_executor.get_num_outputs() == 1
    assert runs_per_sample > 0

    for counter, sample in enumerate(input_data):
        logging.info("Evaluating sample %d", counter)
        aot_executor.get_input(0).copyfrom(sample)
        result = aot_executor.module.time_evaluator("run", session.device, number=runs_per_sample)()
        predicted_label = aot_executor.get_output(0).numpy().argmax()
        runtime = result.mean
        yield predicted_label, runtime
