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
# pylint: disable=missing-docstring

import argparse
import glob
import json
import os
from typing import List

from tqdm import tqdm  # type: ignore
import tvm
from tvm import meta_schedule as ms
from tvm.ir import load_json
from tvm.target import Target


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_cache_dir", type=str, help="Please provide the full path to the extracted tasks."
    )
    parser.add_argument(
        "--candidate_cache_dir",
        type=str,
        help="Please provide the full path to save the sampled candidates.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="nvidia/geforce-rtx-3070",
        help="Please specify the target hardware for tuning.\
                    Note: for generating dataset, the hardware does not need to be present.",
    )
    parser.add_argument(
        "--init_population_size",
        type=int,
        default=256,
        help="The initial population size used in evolutionary search.",
    )
    parser.add_argument(
        "--num_samples_per_task",
        type=int,
        default=400,
        help="The number of samples to gather per tuning task.",
    )
    parser.add_argument(
        "--num_trials_per_iter",
        type=int,
        default=64,
        help="The number of trials per iteration in evolutionary search.",
    )
    parser.add_argument(
        "--max_trials_per_task",
        type=int,
        default=400,
        help="The maximum number of trials per task in evolutionary search.",
    )
    parser.add_argument(
        "--max_retry_per_task",
        type=int,
        default=10,
        help="The maximum number of retry attempts allowed.",
    )
    parser.add_argument(
        "--file_group",
        type=int,
        default=0,
        help="To enable running multiple scripts in parallel, files [idx * 10 : (idx + 1) * 10]\
        in the sorted file list from the given directory will be run.",
    )
    return parser.parse_args()


# pylint: disable=too-many-locals
def sample_candidates(task, task_name, model_name):
    """Randomly sample candidates for a task and save the candidates in the given directory.

    Parameters
    ----------
    task : IRModule
        The initial ir module used for generating the search space.
    task_name : str
        The name of the task.
    model_name : str
        The name of the model.

    Returns
    -------
    None
    """
    candidate_path = os.path.join(
        args.candidate_cache_dir, model_name, task_name + "_candidates.json"
    )
    workload_path = os.path.join(args.candidate_cache_dir, model_name, task_name + "_workload.json")
    database = ms.database.JSONDatabase(
        path_workload=workload_path,
        path_tuning_record=candidate_path,
    )
    sample_init_population = tvm.get_global_func(
        "meta_schedule.SearchStrategyEvolutionarySearchSampleInitPopulation"
    )
    evolve_with_cost_model = tvm.get_global_func(
        "meta_schedule.SearchStrategyEvolutionarySearchEvolveWithCostModel"
    )
    strategy = ms.search_strategy.EvolutionarySearch(
        num_trials_per_iter=args.num_trials_per_iter,
        max_trials_per_task=args.max_trials_per_task,
        init_measured_ratio=0.0,
    )
    target = Target(args.target)
    context = ms.TuneContext(
        mod=task,
        target=target,
        space_generator=ms.space_generator.PostOrderApply(),
        search_strategy=strategy,
        sch_rules=ms.default_config.schedule_rules(None, target),
        postprocs=ms.default_config.postproc(None, target),
        mutator_probs=ms.default_config.mutator_probs(None, target),
        task_name=task_name,
    )
    context.initialize()
    context.pre_tuning(
        context.generate_design_space(),
        database=database,
        cost_model=ms.cost_model.RandomModel(),  # type: ignore
    )

    all_states: List[tvm.tir.Schedule] = []
    num_retry, itr = 0, 0
    states = sample_init_population(strategy, args.init_population_size)
    while len(all_states) < args.num_samples_per_task and num_retry < args.max_retry_per_task:
        states = evolve_with_cost_model(strategy, states, len(states))
        all_states += states
        if len(states) == 0:
            states = sample_init_population(strategy, args.init_population_size)
            num_retry += 1
        else:
            num_retry = 0
        print(f"iter: {itr}, number of states sampled: {len(all_states)}")
        itr += 1
    all_states = all_states[: args.num_samples_per_task]

    workload = ms.database.Workload(context.mod)
    database.commit_workload(context.mod)
    for state in all_states:
        database.commit_tuning_record(ms.database.TuningRecord(state.trace, workload))


args = _parse_args()  # pylint: disable=invalid-name


def main():
    if not os.path.isdir(args.task_cache_dir):
        raise Exception("Please provide a correct task cache dir.")
    try:
        os.makedirs(args.candidate_cache_dir, exist_ok=True)
    except OSError:
        print(f"Directory {args.candidate_cache_dir} cannot be created successfully.")

    task_paths = sorted(glob.glob(os.path.join(args.task_cache_dir, "*.json")))[
        args.file_group * 10 : (args.file_group + 1) * 10
    ]
    print(f"Selected models: {task_paths}")
    for num, task_path in enumerate(task_paths):
        print(f"Processing model {num} ...")
        with open(task_path, "rb") as file:
            tasks = file.readlines()
        model_name = task_path.split("/")[-1][len("relay-") :][: -len("_extracted_tasks.json")]
        os.makedirs(os.path.join(args.candidate_cache_dir, model_name), exist_ok=True)
        for task_str in tqdm(tasks):
            task_name, task_mod = json.loads(task_str)
            task_mod = load_json(json.dumps(task_mod))
            sample_candidates(task_mod, task_name, model_name)


if __name__ == "__main__":
    main()
