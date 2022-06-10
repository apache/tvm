"""
Extract tuning tasks using MetaSchedule, and filter out spatial tasks.
"""

import argparse
import glob
import json
import os
from tqdm import tqdm  # type: ignore

import tvm
from tvm import meta_schedule as ms
from tvm.ir import save_json
from tvm.meta_schedule.testing.relay_workload import _load_cache
from tvm.runtime import load_param_dict


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_cache_dir", type=str, help="Please provide the full path to the model cache dir."
    )
    parser.add_argument(
        "--task_cache_dir", type=str, help="Please provide the full path to save extracted tasks."
    )
    parser.add_argument(
        "--target", type=str, default="cuda", help="Please specify the target hardware for tuning."
    )
    return parser.parse_args()


# pylint: disable=too-many-locals
def extract_and_save_tasks(cache_file):
    """Extract tuning tasks and cache the nonspatial ones in the given directory.

    Parameters
    ----------
    cache_file : str
        The filename of the cached model.

    Returns
    -------
    None

    """

    mod, params_bytearray, _ = _load_cache(args.model_cache_dir, cache_file)
    params = load_param_dict(params_bytearray)
    try:
        extracted_tasks = ms.extract_task_from_relay(mod, target=args.target, params=params)
    except tvm._ffi.base.TVMError:  # pylint: disable=protected-access
        return
    task_cache_path = os.path.join(
        args.task_cache_dir, cache_file.split(".")[0] + "_extracted_tasks.json"
    )
    with open(task_cache_path, "w", encoding="utf8") as file:
        for i, task in enumerate(extracted_tasks):
            subgraph = task.dispatched[0]
            prim_func = subgraph[subgraph.get_global_vars()[0]]
            if not is_spatial(prim_func):
                subgraph_str = save_json(subgraph)
                json_obj = [json.loads(subgraph_str), task.task_name]
                json_str = json.dumps(json_obj)
                assert "\n" not in json_str, "Failed to generate single line string."
                if i == len(extracted_tasks) - 1:
                    file.write(json_str)
                else:
                    file.write(json_str + "\n")


if __name__ == "__main__":
    args = _parse_args()  # pylint: disable=invalid-name
    if not os.path.isdir(args.model_cache_dir):
        raise Exception("Please provide a correct model cache dir.")
    try:
        os.makedirs(args.task_cache_dir, exist_ok=True)
    except OSError as error:
        print(f"Directory {args.task_cache_dir} cannot be created successfully.")

    is_spatial = tvm.get_global_func(  # pylint: disable=invalid-name
        "tir.schedule.IsSpatialPrimFunc"
    )  # pylint: disable=invalid-name
    paths = glob.glob(os.path.join(args.model_cache_dir, "*.json"))  # pylint: disable=invalid-name
    for path in tqdm(paths):
        filename = path.split("/")[-1]
        extract_and_save_tasks(filename)
