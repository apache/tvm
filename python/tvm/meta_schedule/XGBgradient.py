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
A python implementation of Dynamic Gradient Descent Search algorithm
"""
import logging
import os
import random
import time
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from itertools import combinations, product
from math import isqrt, prod
from multiprocessing import Pool

import numpy as np

import tvm
from tvm import meta_schedule as ms

from .builder import Builder
from .cost_model import CostModel
from .database import Database
from .runner import Runner
from tvm.meta_schedule.utils import remove_build_dir

from tvm.runtime import _ffi_node_api


def get_index(array: list, value: int):
    """returns an index if it finds the value"""
    index_map = {element[0]: i for i, element in enumerate(array)}
    return index_map.get(value, -1)


def get_factors(n):
    """
    Return the factors of a given number n as a sorted list.
    """
    factors = []
    sqrt_n = isqrt(n)
    for i in range(1, sqrt_n + 1):
        if n % i == 0:
            factors.append(i)
            j = n // i
            if j != i:
                factors.append(j)
    factors.sort()
    return factors


@staticmethod
def has_sample_instruction(traces) -> bool:
    """
    Returns if a list of traces includes any sample instructions

    Parameters
    ----------
    traces: tvm.schedule.Trace
        The traces to check for sample instructions

    Returns
    -------
    found_sample_inst: bool
        If a sample instruction was found
    """
    # Function could potentially be moved to tvm.schedule.Trace
    sample_instructions = ["SampleComputeLocation", "SampleCategorical", "SamplePerfectTile"]

    for trace in traces:
        for inst in trace.insts:
            if inst.kind.name in sample_instructions:
                return True
    return False


def count_splits(n, k, m):
    """
    统计将整数 n 拆分为 k 个正整数，使得它们的乘积等于 n，
    且最后一个因子小于等于 m 的方案数。
    
    参数：
      n: 待分解的正整数
      k: 分解成的因子个数
      m: 最后一个因子必须 <= m

    返回：
      满足条件的拆分方案数
    """
    if k == 1:
        return 1 if n <= m else 0

    count = 0
    for d in range(1, n + 1):
        if n % d == 0:
            count += count_splits(n // d, k - 1, m)
    return count


@staticmethod
def get_sample_number(traces):
    """
    Returns if a list of traces includes any sample instructions

    Parameters
    ----------
    traces: tvm.schedule.Trace
        The traces to check for sample instructions

    Returns
    -------
    found_sample_inst: bool
        If a sample instruction was found
    """
    sample_num = 1
    for trace in traces:
        for inst in trace.insts:
            if inst.kind.name == "SampleCategorical":
                sample_num *= len(inst.attrs[1])
            elif inst.kind.name == "SamplePerfectTile":
                title_num = inst.attrs[0].value
                inter_most = inst.attrs[1].value
                extend_list = [extend.value for extend in trace.decisions[inst]]
                extend = np.prod(extend_list)
                choices = count_splits(extend, title_num, inter_most)
                sample_num *= choices
    return sample_num


class RecordProcessor:
    """
    A class that processes records and provides methods to extract and modify coordinates.

    Methods:
        get_factors(): Returns the factors of a given number n as a sorted list.
        get_sm_factors(): Returns the shared memory factors of a given number n as a sorted list.
        extract_coordinates(): Extracts coordinates from the SP nodes in the record.
        modify_sp_node(): Modifies the SP nodes in the record to match the new coordinates.
    """

    def __init__(self, record):
        self.record = record
        self.workload = record.workload
        self.record_str = record.as_json()

    @staticmethod
    @lru_cache(maxsize=None)
    def get_factors(n):
        """
        Return the factors of a given number n as a sorted list.
        """
        factors = []
        sqrt_n = isqrt(n)
        for i in range(1, sqrt_n + 1):
            if n % i == 0:
                factors.append(i)
                j = n // i
                if j != i:
                    factors.append(j)
        factors.sort()
        return factors

    def get_sm_factors(self, n):
        """
        Return the shared memory factors of a given number n as a sorted list.
        """
        reg_tile_factors = self.get_factors(n)
        reg_tile_factors_sorted = sorted(reg_tile_factors)
        sm_ts = set()

        for i in range(len(reg_tile_factors_sorted)):
            x = reg_tile_factors_sorted[i]
            max_y = n // x
            j_max = bisect_right(reg_tile_factors_sorted, max_y) - 1
            if j_max < i:
                continue
            for j in range(i, j_max + 1):
                y = reg_tile_factors_sorted[j]
                product = x * y
                if product <= n and n % product == 0:
                    sm_ts.add(product)
        return sorted(sm_ts)

    def extract_coordinates(self):
        """
        Extract coordinates from the split nodes in the record.
        """
        coordinates = []
        self.sample_category = {}

        record_insts = self.record_str[0][0]
        configs = self.record_str[0][1]
        config_dict = {cfg[0]: idx for idx, cfg in enumerate(configs)}

        for counter, config in enumerate(record_insts):
            schedule_name = config[0]
            if schedule_name == "SamplePerfectTile":
                cfg_idx = config_dict.get(counter, -1)
                if cfg_idx == -1:
                    continue
                tile_config = configs[cfg_idx][1]
                coordinates.extend(tile_config[:-1])
            elif schedule_name == "SampleCategorical":
                cfg_idx = config_dict.get(counter, -1)
                if cfg_idx == -1:
                    continue
                coordinates.append(configs[cfg_idx][1])
                self.sample_category[counter] = len(config[2][0])
            elif schedule_name == "Annotate":
                ann_key = config[2]
                if (
                    ann_key == ["meta_schedule.parallel"]
                    or ann_key == ["meta_schedule.vectorize"]
                    or ann_key == ["pragma_auto_unroll_max_step"]
                ):
                    coordinates.append(config[1][1])
                else:
                    continue
        return coordinates

    def modify_multi_level_tiling_node(self, new_coordinates):
        """
        Modify the split nodes in the record to match the new coordinates.
        """
        coord_idx = 0
        record_insts = self.record_str[0][0]
        configs = self.record_str[0][1]
        config_dict = {cfg[0]: idx for idx, cfg in enumerate(configs)}

        for counter, config in enumerate(record_insts):
            schedule_name = config[0]
            if schedule_name == "SamplePerfectTile":
                cfg_idx = config_dict.get(counter, -1)
                if cfg_idx == -1:
                    continue
                tile_config = configs[cfg_idx][1]
                length = len(tile_config) - 1
                new_tile_part = new_coordinates[coord_idx : coord_idx + length]
                original_prod = prod(tile_config)
                new_prod = prod(new_tile_part)
                last_element = original_prod // new_prod if new_prod != 0 else 0
                configs[cfg_idx][1] = list(new_tile_part) + [last_element]
                coord_idx += length
            elif schedule_name == "SampleCategorical":
                cfg_idx = config_dict.get(counter, -1)
                if cfg_idx == -1:
                    continue
                configs[cfg_idx][1] = new_coordinates[coord_idx]
                coord_idx += 1
            elif schedule_name == "Annotate":
                ann_key = config[2]
                if (
                    ann_key == ["meta_schedule.parallel"]
                    or ann_key == ["meta_schedule.vectorize"]
                    or ann_key == ["pragma_auto_unroll_max_step"]
                ):
                    record_insts[counter][1][1] = new_coordinates[coord_idx]
                    coord_idx += 1
                else:
                    continue


def parallel_runner_run(input_group):
    runner = Runner.create("local", max_workers=1)
    runner_result = runner.run(input_group)
    return (
        [v.value for v in runner_result[0].result().run_secs]
        if runner_result[0].result().run_secs
        else [1e10]
    )


def power_of_two(min_value: int, max_value: int) -> list:
    """Return power of two array in interval"""
    return [1 << i for i in range(min_value, max_value + 1)]


def structural_hash(node, map_free_vars=False):
    """Compute structural hash of node

    The structural hash value is recursively defined in the DAG of IRNodes.
    There are two kinds of nodes:

    - Normal node: the hash value is defined by its content and type only.
    - Graph node: each graph node will be assigned a unique index ordered by the
      first occurence during the visit. The hash value of a graph node is
      combined from the hash values of its contents and the index.

    structural_hash is made to be concistent with structural_equal.
    If two nodes are structurally equal to each other,
    then their structural hash (with the same map_free_vars option)
    should be equal to each other as well.

    If the structural hash of two nodes equals to each other,
    then it is highly likely(except for rare hash value collison cases)
    that the two nodes are structurally equal to each other.

    Parameters
    ----------
    node : Object
        The input to be hashed.

    map_free_vars : bool
        If map_free_vars is set to true, we will hash free variables
        by the order of their occurrences. Otherwise, we will hash by
        their in-memory pointer address.

    Return
    ------
    result : int
        The hash result

    See Also
    --------
    structrual_equal
    """
    return _ffi_node_api.StructuralHash(node, map_free_vars)  # type: ignore # pylint: disable=no-member


def assemble_candidates(picks):
    """Assemble a list of candidates from a list of schedules

    Parameters
    ----------
    picks: List[tvm.schedule.Schedule]
        The schedules to turn into MeasureCandidates

    Returns
    -------
    measurement_candidates: List[tvm.meta_schedule.MeasureCandidate]
        The list of MeasureCandidates
    """
    return [
        ms.MeasureCandidate(sch, ms.arg_info.ArgInfo.from_entry_func(sch.mod, remove_preproc=True))
        for sch in picks
    ]


def remove_duplicates_and_measured_schedules(schedules):
    """Remove measured Schedules from a list of Schedules

    Parameters
    ----------
    schedules: List[tvm.schedules.Schedules]
        The list of Schedules to filter

    Returns
    -------
    unique_unmeasured_schedules: List[tvm.schedules.Schedules]
        The list of filtered Schedules
    """
    unique_unmeasured_schedules = []
    added_set: set[int] = set()

    for sch in schedules:
        hashed = structural_hash(sch.mod)
        if hashed not in added_set:
            unique_unmeasured_schedules.append(sch)
            added_set.add(hashed)

    return unique_unmeasured_schedules


class XGBGradientSearchTuner:
    """
    A class that performs dynamic gradient search for auto-scheduling.
    """

    def __init__(
        self,
        task,
        n_start=5,
        init_population_size=8192,
        slide_window_size=3,
        max_trials=1000,
        num_trials_per_iter=80,
        max_tuning_time=120,
        model=None,
        predict_score_threshold_ratio=0.6,
        measure_threshold_ratio=0.6,
        space="post-order-apply",
        target=None,
        task_name=None,
        tmpdir=None,
        warmup_iter=True,
    ):
        """
        Initialize the DynamicGradientSearch object.
        Parameters:
        - task: The task to be optimized.
        - n_start: The number of trials to perform during optimization.
        - init_population_size: The initial size of the model.
        - slide_window_size: The size of the sliding window used for gradient descent.
        - max_trials: The maximum number of trials to perform during optimization.
        - max_tuning_time: The maximum tuning time in seconds.
        - predict_score_threshold_ratio: The threshold ratio for predicted scores.
        - measure_threshold_ratio: The threshold ratio for measured throughput.
        """
        self.task = task
        self.n_start = n_start
        self.init_population_size = init_population_size
        self.n_jobs = os.cpu_count()
        self.slide_window_size = slide_window_size
        self.model = model
        self.measured_throughputs_ = []
        self.count_total_measured = 0
        self.unsuccessful_count = 0
        self.visited = set()
        self.is_cuda = False
        self.max_tuning_time = max_tuning_time
        self.start_time = time.time()
        self.predict_score_threshold_ratio = predict_score_threshold_ratio
        self.measure_threshold_ratio = measure_threshold_ratio
        self.coordinate_set = set()
        self.target = target
        self.context = ms.TuneContext(
            mod=self.task,
            target=self.target,
            space_generator=space,
            search_strategy=ms.search_strategy.EvolutionarySearch(
                population_size=self.init_population_size
            ),
            task_name=task_name,
        )
        self.design_spaces = [space.trace for space in self.context.generate_design_space()]
        self.num_trials_per_iter = num_trials_per_iter
        self.max_trials = max_trials

        self.tmpdir = tmpdir
        self.sample_init_population = tvm.get_global_func(
            "meta_schedule.SearchStrategyEvolutionarySearchSampleInitPopulation"
        )
        self.database = Database.create("json", work_dir=self.tmpdir, module_equality="structural")
        self.context.pre_tuning(
            max_trials=self.max_trials,
            num_trials_per_iter=self.num_trials_per_iter,
            design_spaces=self.context.generate_design_space(),
            database=self.database,
            cost_model=self.model,
        )

        self.builder = Builder.create("local", max_workers=os.cpu_count(), timeout_sec=10.0)
        self.builder_results = []
        self.warmup_iter = warmup_iter

    def get_sample_records(self, number):
        """
        Generate a list of random MeasureInput and MeasureResult pairs.
        Args:
            number: The number of random MeasureInput and MeasureResult pairs to generate.
            task: The task for which the MeasureInput and MeasureResult pairs will be generated.
        Returns:
            tuple: A tuple containing the task, the list of MeasureInput objects
                   and the list of MeasureResult objects.
        """
        logging.debug("Sampling Init Population")
        if number == 1:
            raw_states = [self.context.generate_design_space()[0]]
        else:
            raw_states = self.sample_init_population(self.context.search_strategy, number)
        states = remove_duplicates_and_measured_schedules(raw_states)
        builder_inputs = [ms.builder.BuilderInput(state.mod, self.target) for state in states]
        builder_results = self.builder.build(builder_inputs)
        tuning_records = []
        workload = ms.database.Workload(self.context.mod)
        # Prepare runner inputs
        runner_inputs = []
        valid_indices = []
        for i, res in enumerate(builder_results):
            if res.error_msg:
                continue
            runner_input = ms.runner.RunnerInput(
                res.artifact_path,
                device_type=self.target.kind.name,
                args_info=ms.arg_info.ArgInfo.from_prim_func(self.task),
            )
            runner_inputs.append(runner_input)
            valid_indices.append(i)
        
        if not runner_inputs:
            return

        runner_inputs_2d = list(map(lambda x: [x], runner_inputs))
        with Pool(self.n_jobs) as pool:
            runner_results = pool.map(parallel_runner_run, runner_inputs_2d)

        def process_result(item):
            idx, run_secs = item
            state = states[idx]
            record = ms.database.TuningRecord(
                trace=state.trace,
                workload=workload,
                run_secs=run_secs,
                target=self.target,
                args_info=ms.arg_info.ArgInfo.from_prim_func(self.task),
            )
            return record

        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            tuning_records = list(executor.map(process_result, zip(valid_indices, runner_results)))

        self.count_total_measured += len(raw_states)
        self.builder_results.extend(builder_results)
        return tuning_records

    def dgd_search(self, record):
        """
        Perform the Dynamic Gradient Descent (DGD) search algorithm.
        Utilizes online measurements and proxy model to guide the search process.
        Args:
            record (str): The record string.
            task (Task): The tuning task.
        Returns:
            Tuple: the new base, measured inputs, and measured results.
        """
        logging.debug("NOW EXPLORING 1+2HOP")
        measured_inputs = []
        measured_results = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_1hop = executor.submit(self.get_n_hop_neighbors, record, 1)
            future_2hop = executor.submit(self.get_n_hop_neighbors, record, 2)
            states_1hop_records = future_1hop.result()
            states_2hop_records = future_2hop.result()

        all_neighbors = states_1hop_records + states_2hop_records
        if not all_neighbors:
            return None, None, None

        base_input = record.as_measure_candidate()
        base_result = record.run_secs

        candidate_inputs = [base_input]
        for record in all_neighbors:
            candidate_inputs.append(record.as_measure_candidate())

        candidate_scores = self.model.predict(self.context, candidate_inputs)
        # move to the next base
        new_base, tmp_measured_inputs, tmp_measured_results = self.dgd_move(
            record.run_secs,
            candidate_scores[0],
            candidate_scores[1:],
            all_neighbors,
        )
        if (
            self.count_total_measured >= self.max_trials
            or time.time() - self.start_time >= self.max_tuning_time
        ):
            self.model.update(self.context, measured_inputs, measured_results)
            return new_base, measured_inputs, measured_results

        measured_inputs.extend(tmp_measured_inputs)
        measured_results.extend(tmp_measured_results)

        if not new_base:
            # didn't find new base, then explore 3hop for the current base
            logging.debug("NOW EXPLORING 3HOP")
            all_neighbors = self.get_n_hop_neighbors(record, 3)
            candidate_inputs = [base_input]
            for record in all_neighbors:
                # get all 3 hops and predict/sorted by scores
                candidate_inputs.append(record.as_measure_candidate())

            candidate_scores = self.model.predict(self.context, candidate_inputs)
            new_base, tmp_measured_inputs, tmp_measured_results = self.dgd_move(
                base_result,
                candidate_scores[0],
                candidate_scores[1:],
                all_neighbors,
            )

            if (
                self.count_total_measured >= self.max_trials
                or time.time() - self.start_time >= self.max_tuning_time
            ):
                self.model.update(self.context, tmp_measured_inputs, tmp_measured_results)
                return new_base, measured_inputs, measured_results

            measured_inputs.extend(tmp_measured_inputs)
            measured_results.extend(tmp_measured_results)
            self.model.update(self.context, measured_inputs, measured_results)
        return new_base, measured_inputs, measured_results

    def dgd_move(
        self,
        base_result,
        base_score,
        candidate_scores,
        records,
    ):
        """
        Performs the Dynamic Gradient Descent (DGD) move operation.
        Args:
            base_result (List): The base measurement result.
            base_score (float): The base score used for filtering candidates.
            candidate_inputs (List[MeasureInput]): The list of candidate inputs.
            candidate_scores (List[float]): The list of scores corresponding to the candidates.
        Returns:
            Tuple: the new base, measured inputs, and measured results.
        """
        assert len(candidate_scores) == len(records)
        score_threshold = base_score * self.predict_score_threshold_ratio
        base_cost = np.mean([v.value for v in base_result])
        self.measured_throughputs_.append(1 / base_cost)
        # Filter scores and get the indices of scores that meet the threshold
        filtered_indices = np.where(np.array(candidate_scores) >= score_threshold)[0]

        # Sort the filtered indices based on their scores in descending order
        sorted_indices = filtered_indices[np.argsort(-candidate_scores[filtered_indices])]
        next_base = None
        measured_inputs = []
        measured_results = []
        index_slide = 0
        
        while index_slide < len(sorted_indices) and not next_base:
            slide_window_indices = sorted_indices[index_slide : index_slide + self.slide_window_size]
            selected_records = [records[i] for i in slide_window_indices]
            selected_candidate_inputs = [
                record.as_measure_candidate() for record in selected_records
            ]

            # get the slide window inputs
            slide_window_inputs = [
                ms.builder.BuilderInput(candidate_input.sch.mod, self.target)
                for candidate_input in selected_candidate_inputs
            ]

            # measure the slide window inputs
            builder_results = self.builder.build(slide_window_inputs)
            self.builder_results.extend(builder_results)

            # Prapare runner inputs
            runner_inputs = []
            valid_indices = []
            for i, res in enumerate(builder_results):
                if res.error_msg:
                    continue
                runner_input = ms.runner.RunnerInput(
                    res.artifact_path,
                    device_type=self.target.kind.name,
                    args_info=ms.arg_info.ArgInfo.from_prim_func(self.task),
                )
                runner_inputs.append(runner_input)
                valid_indices.append(i)

            # 如果当前窗口没有有效的 runner 输入，则返回空结果
            if not runner_inputs:
                return None, None, None

            # Run the benchmarks
            runner_inputs_2d = list(map(lambda x: [x], runner_inputs))
            with Pool(self.n_jobs) as pool:
                run_secs_list = pool.map(parallel_runner_run, runner_inputs_2d)

            # Process the results
            def process_update_record(item):
                idx, run_sec = item
                record = ms.database.TuningRecord(
                    selected_records[idx].trace,
                    selected_records[idx].workload,
                    run_sec,
                    selected_records[idx].target,
                    selected_records[idx].args_info,
                )
                return record

            slide_window_costs = np.mean(run_secs_list, axis=1)
            # add to self.measured_throughputs_
            self.measured_throughputs_.extend(1 / np.array(slide_window_costs))

            # threshold
            best_measured = np.max(self.measured_throughputs_)
            measure_threshold = best_measured * self.measure_threshold_ratio

            # early stop
            if (
                1 / np.min(slide_window_costs) < measure_threshold
                and index_slide / len(sorted_indices) > 1 / 3
            ):
                logging.debug("Early stop in current window")
                break

            selected_candidate_inputs = [selected_candidate_inputs[idx] for idx in valid_indices]
            slide_window_results = [ms.runner.RunnerResult(rs, None) for rs in run_secs_list]
            # 并行处理 update_records
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                update_records = list(
                    executor.map(process_update_record, zip(valid_indices, run_secs_list))
                )

            # break after self.max_trials measurements
            if (
                self.count_total_measured + len(slide_window_inputs) >= self.max_trials
                or time.time() - self.start_time >= self.max_tuning_time
            ):
                tmp_size = min(
                    len(slide_window_inputs),
                    self.max_trials - self.count_total_measured,
                )

                tmp_records = update_records[:tmp_size]
                for record in tmp_records:
                    self.database.commit_tuning_record(record)

                self.count_total_measured += tmp_size
                return next_base, measured_inputs, measured_results

            # used for budget control
            self.count_total_measured += len(slide_window_inputs)
            for record in update_records:
                self.database.commit_tuning_record(record)

            # used for updating the model
            measured_inputs.extend(selected_candidate_inputs)
            measured_results.extend(slide_window_results)

            sorted_idx = np.argsort(slide_window_costs)
            # find a better cost to move, add to visited, and avoid re-visit
            for idx in sorted_idx:
                if (
                    slide_window_costs[idx] < base_cost
                    and slide_window_inputs[idx] not in self.visited
                ):
                    next_base = update_records[idx]
                    logging.debug("Found a better base candidate")
                    # add to visited
                    self.visited.add(slide_window_inputs[idx])
                    break
            index_slide += self.slide_window_size
        return next_base, measured_inputs, measured_results

    @lru_cache(maxsize=1024)
    def get_n_hop_neighbors(self, record, n):
        """
        Generate n-hop neighbors for the given record using multi-threading.
        """
        processor = RecordProcessor(record)
        original_coordinates = processor.extract_coordinates()
        dimension = len(original_coordinates)
        self.coordinate_set.add(tuple(original_coordinates))
        record_trace = processor.record.trace.as_json()
        sample_category = processor.sample_category
        # Prepare all combinations of indices and changes for parallel execution
        tasks = []
        for indices in combinations(range(dimension), n):
            for changes in product([-1, 1], repeat=n):
                tasks.append(
                    (
                        original_coordinates,
                        indices,
                        changes,
                        record_trace,
                        sample_category,
                        processor,
                    )
                )

        # Use threading
        neighbors = []
        # Run tasks in parallel
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            results = executor.map(self._process_single_task, tasks)

        for result in results:
            if result is not None:
                neighbors.extend(result)
        return neighbors

    def _process_single_task(self, task):
        """
        Process a single combination of indices and changes to generate neighbors.
        """
        (
            original_coordinates,
            indices,
            changes,
            record_trace,
            sample_category,
            processor,
        ) = task
        new_coordinates = list(original_coordinates)
        valid_change = self._apply_changes(
            new_coordinates, indices, changes, record_trace, sample_category
        )

        if (
            valid_change
            and new_coordinates != original_coordinates
            and tuple(new_coordinates) not in self.coordinate_set
        ):
            processor.modify_multi_level_tiling_node(new_coordinates)
            return [processor.record]
        return None

    def _apply_changes(self, new_coordinates, indices, changes, record_trace, sample_category):
        coord_idx = 0
        for counter, config in enumerate(record_trace[0]):
            schedule_name = config[0]
            if schedule_name == "SamplePerfectTile":
                tile_configs = record_trace[1]
                tile_idx = get_index(tile_configs, counter)
                tile_config = tile_configs[tile_idx][1]
                length = len(tile_config) - 1
                dim_len = prod(tile_config)
                factors = get_factors(dim_len)

                if not self._update_coordinates(
                    new_coordinates,
                    indices,
                    changes,
                    coord_idx,
                    length,
                    factors,
                ):
                    return False

                product_of_dims = prod(new_coordinates[coord_idx : coord_idx + length])
                if product_of_dims > dim_len or dim_len % product_of_dims != 0:
                    return False

                coord_idx += length

            elif schedule_name == "SampleCategorical":
                factors = range(sample_category[counter])

                if not self._update_coordinates(
                    new_coordinates, indices, changes, coord_idx, 1, factors
                ):
                    return False
                coord_idx += 1

            elif schedule_name == "Annotate":
                ann_key = config[2]
                if ann_key == ["meta_schedule.parallel"]:
                    factors = power_of_two(5, 9)
                elif ann_key == ["meta_schedule.vectorize"]:
                    factors = power_of_two(4, 8)
                elif ann_key == ["pragma_auto_unroll_max_step"]:
                    factors = power_of_two(7, 11)
                else:
                    continue
                if not self._update_coordinates(
                    new_coordinates, indices, changes, coord_idx, 1, factors
                ):
                    return False
                coord_idx += 1

        return True

    def _update_coordinates(self, new_coordinates, indices, changes, coord_idx, length, factors):
        for i, change in enumerate(changes):
            idx = indices[i]
            if coord_idx <= idx < coord_idx + length:
                current_value = new_coordinates[idx]
                if current_value in factors:
                    if not self._update_coordinate_with_factor(
                        new_coordinates, idx, change, factors
                    ):
                        return False
                else:
                    self._randomly_pick_factor(new_coordinates, idx, factors)

                # TODO: add cuda constraints
        return True

    def _update_coordinate_with_factor(self, new_coordinates, idx, change, factors):
        factor_index = factors.index(new_coordinates[idx])
        new_factor_index = factor_index + change
        if 0 <= new_factor_index < len(factors):
            new_coordinates[idx] = factors[new_factor_index]
            return True
        return False

    def _randomly_pick_factor(self, new_coordinates, idx, factors):
        random_idx = random.randint(0, len(factors) - 1)
        new_coordinates[idx] = factors[random_idx]

    def XGB_gradient_search(self):
        """
        Perform dynamic gradient search for meta-scheduling.
        Returns:
            None
        """
        # If the samples number is less than 64, we apply brute force search
        total_sample_num = get_sample_number(self.design_spaces)
        if total_sample_num < 64:
            tuning_records = self.get_sample_records(1)
            if not tuning_records:
                print("No valid initial samples found.")
            else:
                candidates = [record.as_measure_candidate() for record in tuning_records]
                results = [
                    ms.runner.RunnerResult(run_secs=record.run_secs, error_msg=None)
                    for record in tuning_records
                ]
                self.model.update(self.context, candidates, results)
                for record in tuning_records:
                    self.database.commit_tuning_record(record)
                    while (
                        record is not None
                        and self.count_total_measured < self.max_trials
                        and time.time() - self.start_time < self.max_tuning_time
                    ):
                        record, _, _ = self.dgd_search(record)
        else:
            # We apply gradient decent with cost model.
            if self.warmup_iter[0] < 1:
                self.warmup_iter[0] += 1
                tuning_records = self.get_sample_records(self.num_trials_per_iter)
                if not tuning_records:
                    logging.debug("No valid initial samples found.")
                    return self.database

                # Update the cost model
                candidates = [record.as_measure_candidate() for record in tuning_records]
                results = [
                    ms.runner.RunnerResult(run_secs=record.run_secs, error_msg=None)
                    for record in tuning_records
                ]

                self.model.update(self.context, candidates, results)
                costs = np.array(
                    [np.mean([v.value for v in res.run_secs]) for res in tuning_records]
                )

                self.measured_throughputs_.extend(1 / np.array(costs))

                topk = min(self.n_start, len(tuning_records))
                topk_indices = np.argsort(costs)[:topk]
                topk_records = [tuning_records[i] for i in topk_indices]

                # use topk as budget now, later will add more options like n trials
                # budget
                for record in topk_records:
                    self.database.commit_tuning_record(record)
                    while (
                        record is not None
                        and self.count_total_measured < self.max_trials
                        and time.time() - self.start_time < self.max_tuning_time
                    ):
                        record, _, _ = self.dgd_search(record)

            else:
                total_sample_num = (
                    self.init_population_size
                    if total_sample_num > self.init_population_size
                    else total_sample_num
                )
                                
                raw_states = self.sample_init_population(
                    self.context.search_strategy, total_sample_num
                )
                candidate_states = remove_duplicates_and_measured_schedules(raw_states)
                candidate_inputs = assemble_candidates(candidate_states)
                # 2. 利用 cost model 对候选方案进行预测打分
                candidates_score = self.model.predict(self.context, candidate_inputs)
                topk = min(self.n_start, len(candidate_inputs))
                topk_indices = np.argsort(-candidates_score)[:topk]
                topk_states = [candidate_states[i] for i in topk_indices]

                # 3. 对 topk 候选进行实际的构建与测量，转换成 tuning record
                builder_inputs = [
                    ms.builder.BuilderInput(state.mod, self.target) for state in topk_states
                ]
                builder_results = self.builder.build(builder_inputs)
                valid_indices = []
                runner_inputs = []
                for i, res in enumerate(builder_results):
                    if res.error_msg:
                        continue
                    runner_input = ms.runner.RunnerInput(
                        res.artifact_path,
                        device_type=self.target.kind.name,
                        args_info=ms.arg_info.ArgInfo.from_prim_func(self.task),
                    )
                    runner_inputs.append(runner_input)
                    valid_indices.append(i)
                if not runner_inputs:
                    logging.debug("No valid candidates found after prediction.")
                    return self.database

                # 使用多进程并行跑测
                runner_inputs_2d = list(map(lambda x: [x], runner_inputs))
                with Pool(self.n_jobs) as pool:
                    run_secs_list = pool.map(parallel_runner_run, runner_inputs_2d)

                topk_records = []
                workload = ms.database.Workload(self.context.mod)
                for idx, run_secs in zip(valid_indices, run_secs_list):
                    record = ms.database.TuningRecord(
                        trace=topk_states[idx].trace,
                        workload=workload,
                        run_secs=run_secs,
                        target=self.target,
                        args_info=ms.arg_info.ArgInfo.from_prim_func(self.task),
                    )
                    topk_records.append(record)

                self.builder_results.extend(builder_results)
                self.count_total_measured += topk
                # 4. 对每个 topk record 作为初始基点，进入动态梯度搜索流程
                for record in topk_records:
                    self.database.commit_tuning_record(record)
                    while (
                        record is not None
                        and self.count_total_measured < self.max_trials
                        and time.time() - self.start_time < self.max_tuning_time
                    ):
                        record, _, _ = self.dgd_search(record)

        for res in self.builder_results:
            if res.artifact_path:
                remove_build_dir(res.artifact_path)

        self.context.post_tuning()
        logging.debug("Finish gradient search")
        return self.database
