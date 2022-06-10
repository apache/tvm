"""
Sample measure candidates for each tuning task by evolutionary search.
"""

import argparse
import glob
import json
import os
from typing import List
from tqdm import tqdm  # type: ignore

import tvm
from tvm import meta_schedule as ms
from tvm.ir import load_json
from tvm.meta_schedule import TuneContext
from tvm.meta_schedule.database import MemoryDatabase, TuningRecord, Workload
from tvm.meta_schedule.search_strategy import EvolutionarySearch
from tvm.meta_schedule.space_generator import PostOrderApply
from tvm.meta_schedule.default_config import _DefaultCUDA, _DefaultLLVM
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

    strategy = EvolutionarySearch(
        num_trials_per_iter=args.num_trials_per_iter,
        max_trials_per_task=args.max_trials_per_task,
    )
    default_config = _DefaultCUDA if args.target != "llvm" else _DefaultLLVM
    context = TuneContext(
        mod=task,
        target=Target(args.target),
        space_generator=PostOrderApply(),
        search_strategy=strategy,
        sch_rules=default_config.schedule_rules(),  # type: ignore
        postprocs=default_config.postprocs(),  # type: ignore
        mutator_probs=default_config.mutator_probs(),  # type: ignore
        task_name=task_name,
    )
    context.initialize()
    spaces = context.generate_design_space()
    strategy.pre_tuning(
        spaces,
        database=MemoryDatabase(),  # type: ignore
        cost_model=ms.cost_model.RandomModel(),  # type: ignore
    )

    all_states: List[tvm.tir.schedule.schedule.Schedule] = []
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

    workload = Workload(context.mod)
    file_path = os.path.join(args.candidate_cache_dir, model_name, task_name + ".json")
    with open(file_path, "w", encoding="utf8") as file:
        for i, state in enumerate(all_states):
            tuning_record = TuningRecord(state.trace, workload)
            json_str = json.dumps(tuning_record.as_json())
            assert "\n" not in json_str, "Failed to generate single line string."
            if i == len(all_states) - 1:
                file.write(json_str)
            else:
                file.write(json_str + "\n")


if __name__ == "__main__":
    args = _parse_args()  # pylint: disable=invalid-name
    if not os.path.isdir(args.task_cache_dir):
        raise Exception("Please provide a correct task cache dir.")
    try:
        os.makedirs(args.candidate_cache_dir, exist_ok=True)
    except OSError as error:
        print(f"Directory {args.candidate_cache_dir} cannot be created successfully.")

    sample_init_population = tvm.get_global_func(  # pylint: disable=invalid-name
        "meta_schedule.SearchStrategyEvolutionarySearchSampleInitPopulation"
    )
    evolve_with_cost_model = tvm.get_global_func(  # pylint: disable=invalid-name
        "meta_schedule.SearchStrategyEvolutionarySearchEvolveWithCostModel"
    )
    task_paths = sorted(  # pylint: disable=invalid-name
        glob.glob(os.path.join(args.task_cache_dir, "*.json"))
    )[  # pylint: disable=invalid-name
        args.file_group * 10 : (args.file_group + 1) * 10
    ]
    print(f"Selected models: {task_paths}")
    for num, task_path in enumerate(task_paths):
        print(f"Processing model {num} ...")
        with open(task_path, "rb") as f:
            tasks = f.readlines()
        model_n = task_path.split("/")[-1][len("relay-") :][: -len("_extracted_tasks.json")]
        os.makedirs(os.path.join(args.candidate_cache_dir, model_n), exist_ok=True)
        for task_str in tqdm(tasks):
            task_mod, task_n = json.loads(task_str)
            task_mod = load_json(json.dumps(task_mod))
            sample_candidates(task_mod, task_n, model_n)
