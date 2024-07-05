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
A python implementation of Dynamic Gradient Descent Search algorithm from ICS'24: Accelerated Auto-Tuning of GPU Kernels for Tensor Computations
"""

import numpy as np
import tvm
from tvm import te, auto_scheduler
from tvm.auto_scheduler.measure_record import load_records
import json
from itertools import combinations, product
from math import isqrt
import os
import random
import time


class RecordProcessor:
    IDX_NODE_NAME = 0
    IDX_STAGE = 1
    IDX_ITER = 2
    IDX_LOOP_EXTENT = 3
    IDX_LENGTHS = 4
    IDX_INNER_TO_OUTER = 5
    IDX_TASK = 0
    IDX_STATE = 1
    IDX_TB = 2
    LENGTH_PAR_DIM = 4
    LENGTH_REDUC = 2

    def __init__(self, record):
        self.record = record
        self.json_str = json.loads(record)

    @staticmethod
    def get_factors(n):
        """
        Return the factors of a given number n as a sorted list.
        """
        factors = set()
        for i in range(1, isqrt(n) + 1):
            if n % i == 0:
                factors.add(i)
                factors.add(n // i)
        return sorted(factors)

    def get_sm_factors(self, n):
        reg_tile_factors = self.get_factors(n)
        sm_ts = set()
        for i in range(len(reg_tile_factors)):
            for j in range(i, len(reg_tile_factors)):
                if (
                    reg_tile_factors[i] * reg_tile_factors[j] <= n
                    and n % (reg_tile_factors[i] * reg_tile_factors[j]) == 0
                ):
                    sm_ts.add(reg_tile_factors[i] * reg_tile_factors[j])
        sm_factors = sorted(list(sm_ts))
        return sm_factors

    def extract_coordinates(self):
        """
        Extract coordinates from the SP nodes in the record.
        """
        coordinates = []
        SP_count = 0

        for each in self.json_str["i"][self.IDX_STATE][1]:
            if (
                each[self.IDX_NODE_NAME] == "SP"
                and len(each[self.IDX_LENGTHS]) == 1
                and each[self.IDX_ITER] == 0
                and SP_count != 0
            ):
                SP_count += 1
                continue
            if each[self.IDX_NODE_NAME] == "SP":
                coordinates.extend(each[self.IDX_LENGTHS])
                SP_count += 1

        return coordinates

    def modify_sp_node(self, new_coordinates):
        """
        Modify the SP nodes in the record to match the new coordinates.
        """
        coord_idx = 0
        SP_count = 0

        for each in self.json_str["i"][self.IDX_STATE][1]:
            if (
                each[self.IDX_NODE_NAME] == "SP"
                and len(each[self.IDX_LENGTHS]) == 1
                and each[self.IDX_ITER] == 0
                and SP_count != 0
            ):
                # if loop extend is the multiple of 2, modify to [2]
                # if each[self.IDX_LOOP_EXTENT] % 2 == 0:
                #     each[self.IDX_LENGTHS] = [2]
                SP_count += 1
                continue
            if each[self.IDX_NODE_NAME] == "SP":
                length = len(each[self.IDX_LENGTHS])
                each[self.IDX_LENGTHS] = new_coordinates[coord_idx : coord_idx + length]
                coord_idx += length
                SP_count += 1
            if each[self.IDX_NODE_NAME] == "PR":
                # aggresive unroll
                each[self.IDX_LOOP_EXTENT] = "auto_unroll_max_step$1024"

        self.record = json.dumps(self.json_str)


class DynamicGradientSearchTuner:
    def __init__(
        self,
        task,
        log_file,
        tune_option,
        n_start=5,
        init_size=64,
        slide_window_size=3,
        max_trials=1000,
        max_tuning_time=120,
        predict_score_threshold_ratio=0.6,
        measure_threshold_ratio=0.6,
    ):
        """
        Initialize the DynamicGradientSearch object.

        Parameters:
        - task: The task to be optimized.
        - log_file: The file path to save the optimization log.
        - n_start: The number of trials to perform during optimization.
        - init_size: The initial size of the model.
        - slide_window_size: The size of the sliding window used for gradient descent.
        - max_trials: The maximum number of trials to perform during optimization.
        - max_tuning_time: The maximum tuning time in seconds.
        - predict_score_threshold_ratio: The threshold ratio for predicted scores.
        - measure_threshold_ratio: The threshold ratio for measured throughput.

        """
        self.task = task
        self.log_file = log_file
        self.n_start = n_start
        self.init_size = init_size
        self.slide_window_size = slide_window_size
        self.model = auto_scheduler.XGBModel(num_warmup_sample=1)
        self.measured_throughputs_ = []
        self.count_total_measured = 0
        self.visited = set()
        self.isCUDA = False
        self.max_trials = max_trials
        self.max_tuning_time = max_tuning_time
        self.start_time = time.time()
        self.predict_score_threshold_ratio = predict_score_threshold_ratio
        self.measure_threshold_ratio = measure_threshold_ratio
        self.Shared_Mem_view = True
        self.coordinate_set = set()

        if tune_option is not None:
            self.runner = tune_option.runner
            self.builder = tune_option.builder
        else:
            """
            tune_option is None, create local runner and builder
            """
            self.runner = auto_scheduler.LocalRunner(timeout=10)
            self.builder = auto_scheduler.LocalBuilder()

        if not os.path.exists(log_file):
            with open(log_file, "w") as fp:
                pass

    def get_sample_records(self, log_file, number, task):
        """
        Generate a list of random MeasureInput and MeasureResult pairs.

        Args:
            log_file (str): The path to the log file where the records will be saved.
            number (int): The number of random MeasureInput and MeasureResult pairs to generate.
            task (Task): The task for which the MeasureInput and MeasureResult pairs will be generated.

        Returns:
            tuple: A tuple containing the task, the list of MeasureInput objects, and the list of MeasureResult objects.
        """
        print("===================================", flush=True)
        print(">>>>  Sampling Init Population <<<<", flush=True)
        print("===================================", flush=True)

        policy = auto_scheduler.SketchPolicy(task, program_cost_model=self.model, verbose=0)
        states = policy.sample_initial_population()
        states = states[: min(number, len(states))]

        inputs = [auto_scheduler.MeasureInput(task, s) for s in states]

        bress = self.builder.build(inputs)
        mress = self.runner.run(inputs, bress)

        with open(log_file, "a") as fp:
            auto_scheduler.save_records(fp.name, inputs, mress)

        self.count_total_measured += len(inputs)

        return inputs, mress

    def DGD_Search(self, log_file, record, task, slide_window_size=3):
        """
        Perform the Dynamic Gradient Descent (DGD) search algorithm.

        Args:
            log_file (str): The path to the log file.
            record (str): The record string.
            task (Task): The tuning task.
            slide_window_size (int, optional): The size of the sliding window. Defaults to 3.

        Returns:
            Tuple: the new base, measured inputs, and measured results.
        """
        print("\n===================================", flush=True)
        print(">>>>   NOW EXPLORING 1+2HOP    <<<<", flush=True)
        measured_inputs = []
        measured_results = []
        base_input, base_result = auto_scheduler.measure_record.load_record_from_string(record)

        states_1hop_record = self.get_n_hop_neighbors(record, 1)
        states_2hop_record = self.get_n_hop_neighbors(record, 2)

        all_neighbors = states_1hop_record + states_2hop_record

        candidate_inputs = [base_input]
        for record_str in all_neighbors:
            # get all 1+2 hops and predict/sorted by scores
            inp, _ = auto_scheduler.measure_record.load_record_from_string(record_str)
            candidate_inputs.append(inp)

        candidate_scores = self.model.predict(task, [x.state for x in candidate_inputs])
        base_score = candidate_scores[0]
        candidate_scores = candidate_scores[1:]
        candidate_inputs = candidate_inputs[1:]

        # move to the next base
        new_base, tmp_measured_inputs, tmp_measured_results = self.DGD_Move(
            log_file,
            base_result,
            base_score,
            candidate_inputs,
            candidate_scores,
            slide_window_size,
        )
        if (
            self.count_total_measured >= self.max_trials
            or time.time() - self.start_time >= self.max_tuning_time
        ):
            return new_base, measured_inputs, measured_results

        measured_inputs.extend(tmp_measured_inputs)
        measured_results.extend(tmp_measured_results)

        if not new_base:
            # didn't find new base, then explore 3hop for the current base
            print("\n===================================", flush=True)
            print(">>>>    NOW EXPLORING 3HOP     <<<<", flush=True)
            all_neighbors = self.get_n_hop_neighbors(record, 3)

            candidate_inputs = [base_input]
            for record_str in all_neighbors:
                # get all 3 hops and predict/sorted by scores
                inp, _ = auto_scheduler.measure_record.load_record_from_string(record_str)
                candidate_inputs.append(inp)

            candidate_scores = self.model.predict(task, [x.state for x in candidate_inputs])
            base_score = candidate_scores[0]

            candidate_scores = candidate_scores[1:]
            candidate_inputs = candidate_inputs[1:]

            new_base, tmp_measured_inputs, tmp_measured_results = self.DGD_Move(
                log_file,
                base_result,
                base_score,
                candidate_inputs,
                candidate_scores,
                slide_window_size,
            )

            if (
                self.count_total_measured >= self.max_trials
                or time.time() - self.start_time >= self.max_tuning_time
            ):
                return new_base, measured_inputs, measured_results

            measured_inputs.extend(tmp_measured_inputs)
            measured_results.extend(tmp_measured_results)

        return new_base, measured_inputs, measured_results

    def DGD_Move(
        self,
        log_file,
        base_result,
        base_score,
        candidate_inputs,
        candidate_scores,
        slide_window_size,
    ):
        """
        Performs the Dynamic Gradient Descent (DGD) move operation.

        Args:
            log_file (str): The path to the log file where the measurement records will be saved.
            base_result (auto_scheduler.MeasureResult): The base measurement result.
            base_score (float): The base score used for filtering candidates.
            candidate_inputs (List[auto_scheduler.MeasureInput]): The list of candidate inputs.
            candidate_scores (List[float]): The list of scores corresponding to the candidate inputs.
            slide_window_size (int): The size of the sliding window used for measurements.

        Returns:
            Tuple: the new base, measured inputs, and measured results.
        """
        assert len(candidate_inputs) == len(candidate_scores)

        score_threshold = base_score * self.predict_score_threshold_ratio
        base_cost = np.mean([v.value for v in base_result.costs])
        global measured_throughputs_
        measured_throughputs_.append(1 / base_cost)

        # sort from large to small
        sorted_indices = np.argsort(candidate_scores)[::-1]

        # Skip candidates with score lower than score threshold
        sorted_indices = [idx for idx in sorted_indices if candidate_scores[idx] >= score_threshold]

        next_base = None
        measured_inputs = []
        measured_results = []

        # apply slide window to the sorted indices, and measure the slide window, until find a better cost neighbor,
        index_slide = 0

        while index_slide < len(sorted_indices) and not next_base:
            if index_slide + slide_window_size > len(sorted_indices):
                slide_window_indices = sorted_indices[index_slide:]
            else:  # slide_window_size <= len(sorted_indices)
                slide_window_indices = sorted_indices[index_slide : index_slide + slide_window_size]

            # get the slide window inputs
            slide_window_inputs = [candidate_inputs[i] for i in slide_window_indices]

            # measure the slide window inputs
            bress = self.builder.build(slide_window_inputs)
            slide_window_results = self.runner.run(slide_window_inputs, bress)

            slide_window_costs = []
            for res in slide_window_results:
                slide_window_costs.append(np.mean([v.value for v in res.costs]))

            # break after self.max_trials measurements
            if (
                self.count_total_measured + len(slide_window_inputs) >= self.max_trials
                or time.time() - self.start_time >= self.max_tuning_time
            ):
                # need to save to the log_file
                tmp_size = min(
                    len(slide_window_inputs),
                    self.max_trials - self.count_total_measured,
                )
                with open(log_file, "a") as fp:
                    tmp_inputs = slide_window_inputs[:tmp_size]
                    tmp_results = slide_window_results[:tmp_size]

                    auto_scheduler.save_records(fp.name, tmp_inputs, tmp_results)

                self.count_total_measured += tmp_size

                return next_base, measured_inputs, measured_results

            # used for budget control
            self.count_total_measured += len(slide_window_inputs)

            # need to save to the log_file
            with open(log_file, "a") as fp:
                auto_scheduler.save_records(fp.name, slide_window_inputs, slide_window_results)

            index_slide += slide_window_size
            # used for updating the model
            measured_inputs.extend(slide_window_inputs)
            measured_results.extend(slide_window_results)

            # add to measured_throughputs_
            for cost in slide_window_costs:
                measured_throughputs_.append(1 / cost)

            # threshold
            best_measured = np.max(measured_throughputs_)
            measure_threshold = best_measured * self.measure_threshold_ratio

            # early stop
            if (
                1 / np.min(slide_window_costs) < measure_threshold
                and index_slide > 3 * slide_window_size
            ):
                print(f">>>>       Early stop         <<<<", flush=True)
                print("===================================", flush=True)
                break

            sorted_idx = np.argsort(slide_window_costs)
            # find a better cost to move, add to visited, and avoid re-visit
            for idx in sorted_idx:
                if (
                    slide_window_costs[idx] < base_cost
                    and slide_window_inputs[idx] not in self.visited
                ):
                    next_base_inp = slide_window_inputs[idx]
                    next_base_res = slide_window_results[idx]
                    next_base = auto_scheduler.measure_record.dump_record_to_string(
                        next_base_inp, next_base_res
                    )
                    print(">>>>      Found a new base     <<<<", flush=True)
                    print("===================================", flush=True)
                    # add to visited
                    self.visited.add(next_base_inp)
                    break

        return next_base, measured_inputs, measured_results

    def get_n_hop_neighbors(self, record, n):
        """
        Generate n-hop neighbors for the given record.
        """
        processor = RecordProcessor(record)
        original_coordinates = processor.extract_coordinates()
        dimension = len(original_coordinates)
        neighbors = []

        self.coordinate_set.add(tuple(original_coordinates))

        # Generate all combinations of coordinates to change
        for indices in combinations(range(dimension), n):
            # Generate all possible changes for the selected coordinates
            for changes in product([-1, 1], repeat=n):
                new_coordinates = original_coordinates[:]
                coord_idx = 0
                valid_change = True  # Add a flag to ensure changes are valid
                SP_count = 0
                for each in processor.json_str["i"][processor.IDX_STATE][1]:
                    if (
                        each[processor.IDX_NODE_NAME] == "SP"
                        and len(each[processor.IDX_LENGTHS]) == 1
                        and each[processor.IDX_ITER] == 0
                        and SP_count != 0
                    ):
                        SP_count += 1
                        continue
                    if each[processor.IDX_NODE_NAME] == "SP":
                        length = len(each[processor.IDX_LENGTHS])
                        dim_len = each[processor.IDX_LOOP_EXTENT]
                        factors = processor.get_factors(dim_len)
                        for i, change in enumerate(changes):
                            idx = indices[i]
                            if (
                                self.Shared_Mem_view
                                and coord_idx <= idx < coord_idx + length
                                and idx - coord_idx == processor.IDX_TB
                                and length == processor.LENGTH_PAR_DIM
                            ):  # tb tile for parallel dimensions
                                sm_factors = processor.get_sm_factors(dim_len)
                                current_lengths = original_coordinates[
                                    coord_idx : coord_idx + length
                                ]
                                sm_tile = np.prod(current_lengths)
                                if sm_tile not in sm_factors:
                                    valid_change = False
                                    break
                                reg_tile = int(sm_tile / new_coordinates[idx])
                                factor_index = sm_factors.index(sm_tile)
                                new_factor_index = factor_index + change
                                if 0 <= new_factor_index < len(sm_factors):
                                    new_sm_tile = sm_factors[new_factor_index]
                                    # check if sm tile > reg tile
                                    if new_sm_tile < reg_tile:
                                        valid_change = False
                                        break
                                    new_coordinates[idx] = int(new_sm_tile / reg_tile)
                                else:
                                    valid_change = False
                                    break
                                if valid_change:
                                    if (
                                        self.isCUDA
                                        and new_coordinates[coord_idx] != 1
                                        and length >= 3
                                    ):
                                        # Force the cuda code has no vthread on parallel dimensions
                                        valid_change = False
                                        break
                            elif coord_idx <= idx < coord_idx + length:
                                current_value = new_coordinates[idx]
                                if current_value in factors:
                                    factor_index = factors.index(current_value)
                                    new_factor_index = factor_index + change
                                    if 0 <= new_factor_index < len(factors):
                                        new_coordinates[idx] = factors[new_factor_index]
                                    else:
                                        valid_change = False
                                        break
                                else:
                                    # random pick a factor, the init tiling might be a non-factor number
                                    random_idx = random.randint(0, len(factors) - 1)
                                    if 0 <= random_idx < len(factors):
                                        random_factor = factors[random_idx]
                                        new_coordinates[idx] = random_factor
                                    else:
                                        valid_change = False
                                        break
                                if valid_change:
                                    if (
                                        self.isCUDA
                                        and new_coordinates[coord_idx] != 1
                                        and length >= 3
                                    ):
                                        # Force the cuda code has no vthread on parallel dimensions
                                        valid_change = False
                                        break
                        if valid_change:
                            product_of_dims = np.prod(
                                new_coordinates[coord_idx : coord_idx + length]
                            )
                            if product_of_dims > dim_len or dim_len % product_of_dims != 0:
                                valid_change = False
                                break
                        coord_idx += length
                        SP_count += 1
                if (
                    valid_change
                    and new_coordinates != original_coordinates
                    and tuple(new_coordinates) not in self.coordinate_set
                ):
                    modified_processor = RecordProcessor(json.dumps(processor.json_str))
                    modified_processor.modify_sp_node(new_coordinates)
                    neighbors.append(modified_processor.record)

        return neighbors

    def dynamic_gradient_search(self):
        """
        Perform dynamic gradient search for auto-scheduling.

        Returns:
            None
        """
        log_file = self.log_file
        task = self.task
        init_size = self.init_size
        n_start = self.n_start
        slide_window_size = self.slide_window_size

        if "cuda" in str(task.target):
            """
            Start DGD_Search for CUDA, apply loop permutation view for CUDA
            """
            self.isCUDA = True
            hardware_params = auto_scheduler.HardwareParams(
                target=task.target, max_vthread_extent=1
            )
            new_task = auto_scheduler.SearchTask(
                workload_key=task.workload_key,
                target=task.target,
                hardware_params=hardware_params,
                layout_rewrite_option=task.layout_rewrite_option,
                task_inputs=list(task.task_input_names),
            )
            task = new_task
            self.task = task

        # use 1/exe_time as the throughput
        global measured_throughputs_
        measured_throughputs_ = []

        inputs, results = self.get_sample_records(log_file, init_size, task)

        self.model.update(inputs, results)

        list_costs = []
        records = []
        topk = min(n_start, len(inputs))
        for inp, res in zip(inputs, results):
            record_str = auto_scheduler.measure_record.dump_record_to_string(inp, res)
            costs = [v.value for v in res.costs]
            cost = np.mean(costs)
            list_costs.append(cost)
            records.append(record_str)
            measured_throughputs_.append(1 / cost)

        topk_indices = np.argsort(list_costs)[:topk]
        topk_records = [records[i] for i in topk_indices]

        # use topk as budget now, later will add more options like ntrials budget
        for record in topk_records:
            while (
                record != None
                and self.count_total_measured < self.max_trials
                and time.time() - self.start_time < self.max_tuning_time
            ):
                record, measured_inputs, measured_results = self.DGD_Search(
                    log_file, record, task, slide_window_size
                )

                # update the model with the new results
                self.model.update(measured_inputs, measured_results)

            if (
                self.count_total_measured >= self.max_trials
                or time.time() - self.start_time >= self.max_tuning_time
            ):
                break

        while (
            self.count_total_measured < self.max_trials
            and time.time() - self.start_time < self.max_tuning_time
        ):
            # keep sampling
            inputs, results = self.get_sample_records(log_file, 1, task)
            record = auto_scheduler.measure_record.dump_record_to_string(inputs[0], results[0])

            self.model.update(inputs, results)

            while record != None:
                record, measured_inputs, measured_results = self.DGD_Search(
                    log_file, record, task, slide_window_size
                )

                # update the model with the new results
                self.model.update(measured_inputs, measured_results)

        print("===================================", flush=True)
        print(">>>>          Done             <<<<", flush=True)
        print("===================================", flush=True)
        return
