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
"""Relax Tuning Pass API default functions"""
from typing import Dict, List, Optional
import sys
import itertools
import logging
import numpy as np  # type: ignore

import tvm
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext, Pass
from tvm import meta_schedule
from tvm.meta_schedule.arg_info import TensorInfo
from tvm.meta_schedule.builder import BuilderInput, LocalBuilder
from tvm.meta_schedule.utils import get_global_func_with_default_on_worker
from tvm.meta_schedule.runner import (
    EvaluatorConfig,
    LocalRunner,
    RunnerInput,
)
from tvm._ffi.registry import register_func
from .primitives import Knob, Trace

logger = logging.getLogger("TuningAPI")  # pylint: disable=invalid-name

# Default transform func that returns original IRModule.
@tvm.register_func("relax.tuning_api.Choice.default_transform_func")
def default_transform_func(mod):
    return mod


# Default constraint func that always returns true.
@tvm.register_func("relax.tuning_api.Choice.default_constr_func")
def default_constr_func(mod: IRModule) -> bool:  # pylint: disable=unused-argument
    return True


@register_func("relax.tuning_api.default_generate_candidate")
def default_generate_candidate(
    knobs: List[Knob], trace: Trace, eval_passes: Optional[List[Pass]] = None
) -> List[Trace]:
    """
    Default function to generate the search space for a given trace by using registered choices.
    This function simply expands candidate space as long as the knob's constraint satisfies.
    To reduce the search space, a developer may expand each choice with smart search method.
    (e.g., genetic search, multi-armed bandit)
    Note, each pass generates candidates without worrying about the interaction with other passes.
    i.e., it only uses its incoming trace/IRModule and Choices for candidate generation.
    This will help alleviating the complexity of joint-optimization significantly.
    - consideration of interaction between optimizations has known to be extremely difficult.

    Parameters
    ----------
    knobs : List[Knob]
        List of Knobs to consider to generate candidate for input trace.
    trace: Trace
        Input trace.
    eval_passes: Optional[List[Pass]]
        List of passes to consider to evaluate each candidate.
        This will enable joint-optimization.

    Return
    ----------
    candidates: List[Trace]
        List of candidate traces
    """

    candidates = [trace]
    # Iterate over every decision
    for knob in knobs:
        num = len(candidates)
        for _ in range(num):
            cur_trace = candidates.pop(0)
            for decision in knob.choices.keys():
                choice = knob.choices[decision]
                # Generate new candidate when this condition satisfies.
                if choice.check_constr(cur_trace.out_mod):
                    new_trace = cur_trace.deepcopy()
                    new_trace.add(knob, decision)
                    candidates.append(new_trace)

    # Expand candidates by using eval passes if provided. This will enable joint-optimization.
    if eval_passes:
        candidates = default_consider_eval_passes(candidates, eval_passes)
    return candidates


@register_func("relax.tuning_api.default_consider_eval_passes")
def default_consider_eval_passes(
    init_candidates: List[Trace], eval_passes: Optional[List[Pass]] = None
) -> List[Trace]:
    """
    Default function to update traces with eval passes.
    It visits each eval_pass in dfs order in transform.Sequential() and
    returns the best possible candidate trace for each candidate.

    Parameters
    ----------
    init_candidates: List[Trace]
        Initial candidates
    eval_passes: Optional[List[Pass]]
        List of passes to consider to evaluate each candidate.
        This will enable joint-optimization.
    Return
    ----------
    candidates: List[Trace]
        List of candidate traces
    """
    if not eval_passes:
        return init_candidates

    eval_passes = list(eval_passes) if not isinstance(eval_passes, list) else eval_passes
    ctx = PassContext.current()
    candidates = []

    for trace in init_candidates:
        ctx.push_trace(trace)
        tvm.transform.Sequential(eval_passes)(trace.out_mod)
        new_trace = ctx.pop_trace()
        # A new trace contains the best decisions in eval_passes
        candidates.append(new_trace)

    return candidates


@register_func("relax.tuning_api.default_evaluate")
def default_evaluate(
    candidates: List[Trace],
    target_str: str,
    params: Optional[Dict[str, np.ndarray]] = None,
    builder: Optional[meta_schedule.builder.Builder] = None,
    runner: Optional[meta_schedule.runner.Runner] = None,
) -> None:
    """
    Default function to evaluate a set of candidate traces by using MetaSchedule builder/runner.

    Parameters
    ----------
    candidates: List[Trace]
        List of traces to evaluate.
    target_str: str,
        Compilation target (e.g., llvm, cuda).
    params: Optional[Dict[str, np.ndarray]]
        Params to bind.
    builder: Optional[meta_schedule.builder.Builder]
        builder function. If not provided, default local builder will be used.
    runner: Optional[meta_schedule.runner.Runner]
        runner function. If not provided, default local runner will be used.
    """

    ctx = PassContext.current()
    target = tvm.target.Target(target_str)
    database = PassContext.current().get_tuning_api_database()
    # Setup default local builder if not provided
    if builder is None:

        def relax_build(
            mod: IRModule,
            target: tvm.target.Target,
            params: Optional[Dict[str, np.ndarray]],
        ):
            if params:
                mod = tvm.relax.transform.BindParams("main", params)(mod)
            relax_exec = tvm.relax.build(mod, target)
            return relax_exec.mod

        builder = LocalBuilder(f_build=relax_build)

    # Setup default local runner if not provided
    if runner is None:

        def relax_eval_func(rt_mod, device, evaluator_config, repeated_args):
            relax_exec = tvm.relax.Executable(rt_mod)
            relax_vm = tvm.relax.VirtualMachine(relax_exec, device=device)

            evaluator = relax_vm.module.time_evaluator(
                func_name="main",
                dev=device,
                number=evaluator_config.number,
                repeat=evaluator_config.repeat,
                min_repeat_ms=evaluator_config.min_repeat_ms,
            )
            repeated_costs: List[List[float]] = []
            for args in repeated_args:
                profile_result = evaluator(*args)
                repeated_costs.append(profile_result.results)

            costs = [float(cost) for cost in itertools.chain.from_iterable(repeated_costs)]

            return costs

        runner = LocalRunner(
            evaluator_config=EvaluatorConfig(
                number=3, repeat=5, min_repeat_ms=100, enable_cpu_cache_flush=False
            ),
            f_run_evaluator=relax_eval_func,
        )

    # set up clean up function
    f_clean_build = get_global_func_with_default_on_worker("meta_schedule.remove_build_dir", None)
    assert f_clean_build

    # Keep track of number of evaluations (mostly for the debugging purpose)
    num_evals = 0
    # Evaluation
    for candidate in candidates:
        # If this candidate is already evaluated, skip the measurement
        if candidate.perf != -1:
            continue

        # Evaluate candidates
        num_evals += 1
        mod = candidate.out_mod
        workload = database.commit_workload(mod)

        # If this workload and target pair has measured before, fetch its data.
        if database.has_measurement_record(workload, target):
            run_secs = database.get_measurement_record(workload, target)
        # Otherwise, measure it.
        else:
            # Build candidate
            (builder_result,) = builder.build([BuilderInput(mod, target, params)])

            if builder_result.artifact_path is None:
                # Build error
                # Assign the worst performance and move on to the next candidate.
                logger.warning(builder_result.error_msg)
                run_secs = [1e100]
            else:
                # If build passes, set up runner input and measure the performance.
                args_info = [
                    TensorInfo(
                        shape=[int(i) for i in p.struct_info.shape], dtype=p.struct_info.dtype
                    )
                    for p in mod["main"].params
                ]  # convert list[Var] to list[TensorInfo]
                runner_input = RunnerInput(
                    builder_result.artifact_path, target_str, args_info=args_info
                )
                (runner_future,) = runner.run([runner_input])
                runner_result = runner_future.result()

                run_secs = runner_result.run_secs
                # Runtime error
                # Assign the worst performance and move on to the next candidate.
                if runner_result.error_msg is not None:
                    logger.warning(runner_result.error_msg)
                    run_secs = [1e100]

                database.commit_measurement_record(workload, target, run_secs)

            # Clean up the artifact
            f_clean_build(builder_result.artifact_path)

        # For valid measurments, compute the average and update the trace performance.
        perfs = []
        for result in run_secs:
            if isinstance(result, tvm.tir.FloatImm):
                result = result.value
            assert isinstance(result, float)
            assert result >= 0.0
            perfs.append(result)

        # Store the evaluation result
        candidate.set_perf(np.mean(perfs))

    ctx.inc_num_evals(num_evals)


def select_best_candidate(candidates: List[Trace]) -> Trace:
    """
    Select the best trace.

    Parameters
    ----------
    candidates: List[Trace]
        Candidate traces

    Return
    ----------
    best_trace: Trace
        Trace with the best performance
    """
    best_perf, best_trace = sys.maxsize, None
    for candidate in candidates:
        avg = candidate.perf
        # Select best one
        if best_perf > avg:
            best_perf = avg
            best_trace = candidate
    return best_trace
