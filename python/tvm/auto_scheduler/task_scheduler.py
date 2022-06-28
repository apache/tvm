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
# pylint: disable=invalid-name

""" The task scheduler that allocates the time resources when tuning multiple tasks together

The details of the "gradient" strategy below can be found in the section 6 of this paper:
L. Zheng, C. Jia, M. Sun, Z. Wu, C. Yu, et al. "Ansor : Generating High-Performance Tensor
Programs for Deep Learning." (OSDI 2020).
"""
import os
import time
import math
import logging

import numpy as np

from .search_policy import SearchPolicy, SketchPolicy, PreloadMeasuredStates
from .cost_model import RandomModel, XGBModel
from .utils import array_mean
from .measure import ProgramMeasurer
from .measure_record import RecordReader
from . import _ffi_api

logger = logging.getLogger("auto_scheduler")


def make_search_policies(
    search_policy,
    search_policy_params,
    tasks,
    num_measures_per_round,
    verbose,
    load_model_file=None,
    load_log_file=None,
    adaptive_training=False,
):
    """Make a list of search policies for a list of search tasks.
    It creates one policy per task.

    Parameters
    ----------
    search_policy: Union[str, List[SearchPolicy]]
        The name of search policy.
    search_policy_params: Dict[str, Any]]
        The parameters of the search policy.
    tasks: List[SearchTask]
        The list of all tasks
    num_measures_per_round: int
        The number of schedules to be measured at each search round.
        This should be the same as `TuningOptions.num_measures_per_round`
    verbose: int
        The verbosity level. 0 for silent.
    load_model_file: Optional[str]
        Load pre-trained model from this file. If this is None, the cost model will
        be trained from scratch.
    load_log_file: Optional[str]
        Load measurement records from this file. If it is not None, the status of the
        task scheduler, search policies and cost models will be restored according to this file.
    adaptive_training: bool = False
        Option used by XGBModel to reduce the model training frequency when there're too
        many logs.

    Returns
    -------
    policies: List[SearchPolicy]
        The list of search policies
    """
    if search_policy == "default":
        search_policy = "sketch.xgb"

    if isinstance(search_policy, str):
        policy_type, model_type = search_policy.split(".")
        if model_type == "xgb":
            cost_model = XGBModel(
                num_warmup_sample=len(tasks) * num_measures_per_round,
                model_file=load_model_file,
                adaptive_training=adaptive_training,
            )
            if load_model_file and os.path.isfile(load_model_file):
                logger.info("TaskScheduler: Load pretrained model...")
                cost_model.load(load_model_file)
            elif load_log_file:
                logger.info("TaskScheduler: Reload measured states and train the model...")
                cost_model.update_from_file(load_log_file)
        elif model_type == "random":
            cost_model = RandomModel()
        else:
            raise ValueError("Invalid search policy: " + search_policy)

        if policy_type == "sketch":
            if load_log_file:
                # use the log file to restore the status of search policies.
                init_search_callbacks = [PreloadMeasuredStates(load_log_file)]
            else:
                init_search_callbacks = None
            search_policies = [
                SketchPolicy(
                    task,
                    cost_model,
                    params=search_policy_params,
                    verbose=verbose,
                    init_search_callbacks=init_search_callbacks,
                )
                for task in tasks
            ]
        else:
            raise ValueError("Invalid search policy: " + search_policy)
    else:
        # check type
        assert isinstance(search_policy, (tuple, list))
        for item in search_policy:
            assert isinstance(item, SearchPolicy)
        search_policies = search_policy

    return search_policies


def derive_similarity_tag(dag, log_base=1.618):
    """Derive the tag for similarity check from one computational DAG.
    The DAGs with the same tag are considered as similar tasks.

    The tag format is <op1-tag>_<op2-tag> ... <log(flop)>.

    If the tag is "", then the task is not considered to be similar to any other tasks.

    Parameters
    ----------
    dag: ComputeDAG
        The input computational DAG
    log_base: float = 1.618
        The base of log to normalize FLOPS

    Returns
    -------
    tag: str
        The tag of this computational DAG.
    """
    ret = ""
    for op in dag.ops:
        tag = op.attrs.get("auto_scheduler_task_scheduler_tag", None)
        if tag:
            ret += op.attrs["auto_scheduler_task_scheduler_tag"] + "_"
    if ret:
        ret += "%d" % int(math.log(dag.flop_ct + 1, log_base))
    return ret


class TaskScheduler:
    """
    Allocate the time resources when tuning multiple tasks together.
    This implements two strategies: "round-robin" and "gradient".

    Parameters
    ----------
    tasks: List[SearchTask]
        All tasks to tune
    task_weights: Optional[List[float]]
        The weights of tasks.
        If provided, the task scheduler will set the objective function to
        sum(weight[t] * latency[t]), where weight[t] is the weight of a task
        and the lantecy[t] is the lantecy of the task.
        If not provided, the task scheduer will assign equal weights to all
        tasks (i.e., the objective function is sum(latency[t])).
    objective_func: Optional[Callable[List[float] -> float]]
        The objective function to be minimized.
        The objective function accepts the current latencies of all tasks and returns the
        objective.
        If not provided, the objective is the weighted sum of the latencies of all tasks.
    strategy: str = "gradient"
        The scheduling strategy.
        "round-robin": Tune tasks in round robin order.
        "gradient" : Tune tasks with gradient descent.
    load_model_file: Optional[str]
        Load pre-trained model from this file. If this is None, the cost model will
        be trained from scratch.
    load_log_file: Optional[str]
        Load measurement records from this file. If it is not None, the status of the
        task scheduler, search policies and cost models will be restored according to this file.
    verbose: int = 1
        The level of verbosity. 0 means silent.
    alpha: float = 0.2
        The parameter used for 'gradient' strategy
    beta: float = 2
        The parameter used for 'gradient' strategy
    backward_window_size: int = 3
        The parameter used for 'gradient' strategy
    callbacks: Optional[List[TaskSchedulerCallback]]
        The task scheduler callbacks that will be called before and after tuning a task.
        If None, PrintTableInfo and LogEstimatedLatency callback will be used.
    """

    def __init__(
        self,
        tasks,
        task_weights=None,
        objective_func=None,
        strategy="gradient",
        load_model_file: str = None,
        load_log_file: str = None,
        alpha: float = 0.2,
        beta: float = 2,
        gamma: float = 0.5,
        backward_window_size: int = 3,
        callbacks=None,
    ):
        self.tasks = tasks
        if objective_func:  # use custom objective function
            self.objective_func = objective_func
        else:  # use weighted sum
            if task_weights:
                self.objective_func = lambda costs: sum(c * w for c, w in zip(costs, task_weights))
            else:
                self.objective_func = sum

        self.strategy = strategy
        self.load_log_file = load_log_file
        self.load_model_file = load_model_file
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.backward_window_size = backward_window_size
        self.callbacks = (
            callbacks
            if callbacks is not None
            else [PrintTableInfo(), LogEstimatedLatency("total_latency.tsv")]
        )

        assert len(self.tasks) != 0, "No tasks"
        assert self.strategy in ["round-robin", "gradient"]

        # task_cts[i] saves how many times task i is tuned
        self.task_cts = [0 for _ in range(len(self.tasks))]

        # task_best_cts[i] saves the round task i found the best latency
        self.task_best_cts = [0 for _ in range(len(self.tasks))]

        # task_costs_history[i] saves the latency history of task i
        self.task_costs_history = [[] for _ in range(len(self.tasks))]

        # best_costs[i] saves the best latency of task i
        self.best_costs = 1e10 * np.ones(len(self.tasks))
        self.cur_score = self._compute_score(self.best_costs)

        self.tune_option = self.measurer = self.search_policies = None
        self.ct = self.best_ct = self.best_score = self.tic = None
        self.num_measures_per_round = None
        self.dead_tasks = set()

        # Build similarity groups
        self.task_tags = []  # task_id -> tag
        self.tag_to_group_id = {}  # tag -> group_id
        self.group_task_ids = []  # group_id -> all task ids in this group
        self.flop_cts = []  # task_id -> the number of floating ops
        for i, task in enumerate(self.tasks):
            tag = derive_similarity_tag(task.compute_dag)
            self.task_tags.append(tag)
            self.flop_cts.append(task.compute_dag.flop_ct)
            if not tag:
                continue

            if tag not in self.tag_to_group_id:
                self.tag_to_group_id[tag] = len(self.tag_to_group_id)
                self.group_task_ids.append([])
            self.group_task_ids[self.tag_to_group_id[tag]].append(i)

    def tune(
        self,
        tune_option,
        search_policy="default",
        search_policy_params=None,
        adaptive_training=False,
        per_task_early_stopping=None,
    ):
        """Tune a batch of tasks together.

        Parameters
        ----------
        tune_option: TuningOptions
            The tuning options applied to all tasks.
        search_policy: : Union[str, List[SearchPolicy]] = "default"
            The list of search policies.
            If it is str,
            "default" for the default policy (SketchPolicy + XGBModel),
            "sketch.xgb" for SketchPolicy + XGBModel,
            "sketch.random" for SketchPolicy + RandomModel.
        search_policy_params : Optional[Dict[str, Any]]
            The parameters of the search policy
        adaptive_training : bool = False
            Option used by XGBModel to reduce the model training frequency when there're
            too many logs.
        per_task_early_stopping : Optional[int]
            Stop tuning a task early if getting no improvement after n measurements.
        """
        # init members
        self.tune_option = tune_option
        self.early_stopping_all = (
            1e20 if tune_option.early_stopping < 0 else tune_option.early_stopping
        )
        self.early_stopping_task = (
            1e20 if per_task_early_stopping is None else per_task_early_stopping
        )

        self.measurer = ProgramMeasurer(
            tune_option.builder,
            tune_option.runner,
            tune_option.measure_callbacks,
            tune_option.verbose,
        )
        self.ct = self.best_ct = 0
        self.tic = time.time()

        # reset num_measures_per_round to make sure every task is tuned at least once
        self.num_measures_per_round = min(
            tune_option.num_measures_per_round, tune_option.num_measure_trials // len(self.tasks)
        )
        if self.num_measures_per_round <= 0:
            raise ValueError(
                "num_measure_trials is too small. Please set it to a higher value."
                f"It should be at least {len(self.tasks)} for this model."
            )

        # restore the status of the task scheduler from a log file
        if self.load_log_file:
            self._restore_status(self.load_log_file, self.num_measures_per_round)

        # make one search policy for one task
        self.search_policies = make_search_policies(
            search_policy,
            search_policy_params,
            self.tasks,
            self.num_measures_per_round,
            tune_option.verbose,
            self.load_model_file,
            self.load_log_file,
            adaptive_training,
        )

        # do a round robin first to warm up
        for idx in range(len(self.tasks)):
            # skip warming up this task if it has been tuned before (restored from the log file)
            if not self.task_cts[idx]:
                self._tune_task(idx)
        self.best_ct = self.ct
        self.best_score = self.cur_score

        # use the specific strategy to choose workload to tune
        task_idx = -1
        while self.ct < tune_option.num_measure_trials and len(self.dead_tasks) < len(self.tasks):
            if self.strategy == "round-robin":
                task_idx = (task_idx + 1) % len(self.tasks)
                while task_idx in self.dead_tasks:
                    task_idx = (task_idx + 1) % len(self.tasks)
            elif self.strategy == "gradient":
                gradients = []
                for i in range(len(self.tasks)):
                    if i in self.dead_tasks:
                        gradients.append(0)
                        continue

                    # compute gradient from chain rule : (delta f / delta g_i)
                    delta = 1e-4
                    new_costs = list(self.best_costs)
                    new_costs[i] -= delta
                    chain_grad = (
                        self._compute_score(self.best_costs) - self._compute_score(new_costs)
                    ) / delta

                    # compute (g_i(t_i) - g(t_i - \Delta t)) / (\Delta t)
                    if (
                        self.task_cts[i] - 1 < len(self.task_costs_history[i])
                        and self.task_cts[i] - 1 - self.backward_window_size >= 0
                    ):
                        backward_grad = (
                            self.task_costs_history[i][self.task_cts[i] - 1]
                            - self.task_costs_history[i][
                                self.task_cts[i] - 1 - self.backward_window_size
                            ]
                        ) / self.backward_window_size
                    else:
                        backward_grad = 0

                    # compute (g_i(t_i + \Delta t) - g(t_i)) / (\Delta t)
                    g_next_1 = self.best_costs[i] - (self.best_costs[i] / self.task_cts[i])

                    g_next_2 = self.beta * 1e30
                    group_id = self.tag_to_group_id.get(self.task_tags[i], None)
                    if group_id is not None and len(self.group_task_ids[group_id]) > 1:
                        best_flops = max(
                            [
                                self.flop_cts[j] / self.best_costs[j]
                                for j in self.group_task_ids[group_id]
                            ]
                        )
                        g_next_2 = self.beta * self.flop_cts[i] / best_flops

                    g_next = min(g_next_1, g_next_2)
                    forward_grad = g_next - self.best_costs[i]

                    # combine all grads
                    grad = chain_grad * (
                        self.alpha * backward_grad + (1 - self.alpha) * forward_grad
                    )
                    assert grad <= 0
                    gradients.append(grad)

                if max(gradients) == min(gradients):
                    task_idx = np.random.choice(len(gradients))
                else:
                    task_idx = np.argmin(gradients)
            else:
                raise ValueError("Invalid strategy: " + self.strategy)

            self._tune_task(task_idx)
            self._adjust_similarity_group(task_idx)

            if self.cur_score < self.best_score:
                self.best_score = self.cur_score
                self.best_ct = self.ct
            elif self.ct - self.best_ct >= self.early_stopping_all and all(
                cost < 1e9 for cost in self.best_costs
            ):
                if self.tune_option.verbose >= 1:
                    print(
                        "Stop early since no performance improvement in the last "
                        + str(self.early_stopping_all)
                        + " measurement trials."
                    )
                break

    def _tune_task(self, task_idx):
        """Tune the select task for one round"""

        # Run pre-tune callbacks
        for callback in self.callbacks:
            callback.pre_tune(self, task_idx)

        measure_inputs, measure_results = self.search_policies[task_idx].continue_search_one_round(
            self.num_measures_per_round, self.measurer
        )

        self.task_cts[task_idx] += 1

        for res in measure_results:
            cost = array_mean(res.costs)
            if cost < self.best_costs[task_idx]:
                self.task_best_cts[task_idx] = self.task_cts[task_idx]
                self.best_costs[task_idx] = cost

        # Stop tuning this task in the rest of the process if its search space has been
        # fully explored or it has no improvement for a long while.
        no_change_trials = (
            self.task_cts[task_idx] - self.task_best_cts[task_idx]
        ) * self.num_measures_per_round
        if len(measure_inputs) == 0 or no_change_trials > self.early_stopping_task:
            self.dead_tasks.add(task_idx)

        self.task_costs_history[task_idx].append(self.best_costs[task_idx])

        self.ct += len(measure_inputs)
        self.cur_score = self._compute_score(self.best_costs)

        # Run post-tune callbacks
        for callback in self.callbacks:
            callback.post_tune(self, task_idx)

    def _compute_score(self, costs):
        """compute the objective function"""
        # Make sure to return float.
        score = self.objective_func(costs)
        return score.value if hasattr(score, "value") else score

    def _adjust_similarity_group(self, task_idx):
        """adjust the similarity group for the selected task"""
        group_id = self.tag_to_group_id.get(self.task_tags[task_idx], None)
        if group_id is None or len(self.group_task_ids[group_id]) <= 1:
            return

        group_ids = self.group_task_ids[group_id]
        best_group_flops = max([self.flop_cts[j] / self.best_costs[j] for j in group_ids])
        cur_flops = self.flop_cts[task_idx] / self.best_costs[task_idx]

        # if we tune a task for many times but it still cannot achieve
        # a similar speed to the fastest one in its group, this means this task
        # is actually not similar to other tasks in its group.
        # So we will remove it from its original group.
        if cur_flops < best_group_flops / self.beta and self.task_cts[task_idx] > 5 + max(
            self.task_cts[j] for j in group_ids if j != task_idx
        ):
            self.task_tags[task_idx] = None
            group_ids.remove(task_idx)

    def _restore_status(self, log_file, num_measures_per_round):
        """restore task_cts and best_costs from a log file"""
        str_target = str(self.tasks[0].target)
        workload_key_to_task_id = {t.workload_key: i for i, t in enumerate(self.tasks)}
        total_ct = -1

        for total_ct, (inp, res) in enumerate(RecordReader(log_file)):
            if str(inp.task.target) != str_target:
                continue
            task_idx = workload_key_to_task_id.get(inp.task.workload_key, None)
            if task_idx is None:
                continue

            self.task_cts[task_idx] += 1

            if res.error_no == 0:
                cost = array_mean(res.costs)
                if cost < self.best_costs[task_idx]:
                    self.best_costs[task_idx] = cost
                    self.task_best_cts[task_idx] = self.task_cts[task_idx]

        for idx in range(len(self.tasks)):
            if self.task_cts[idx] - self.task_best_cts[idx] > self.early_stopping_task:
                self.dead_tasks.add(idx)

            # The computation of taks_cts is just an estimation.
            # The estimation may not be accurate if the log file is changed externally or
            # `num_measures_per_round` is different from the last tuning.
            self.task_cts[idx] = int(self.task_cts[idx] / num_measures_per_round + 0.5)
            self.task_best_cts[idx] = int(self.task_best_cts[idx] / num_measures_per_round + 0.5)
            self.task_costs_history[idx].append(self.best_costs[idx])

        self.cur_score = self._compute_score(self.best_costs)

        logger.info("TaskScheduler: Loaded %d measurement records from %s", total_ct + 1, log_file)


class TaskSchedulerCallback:
    """The base class of task scheduler callback functions."""

    def pre_tune(self, task_scheduler, task_id):
        """The callback before tuning each task.

        Parameters
        ----------
        task_scheduler: TaskScheduler
            The task scheduler.
        task_id: int
            The task ID going to be tuned.
        """
        # Do nothing by default

    def post_tune(self, task_scheduler, task_id):
        """The callback after tuning each task.

        Parameters
        ----------
        task_scheduler: TaskScheduler
            The task scheduler.
        task_id: int
            The task ID be tuned.
        """
        # Do nothing by default


class PrintTableInfo(TaskSchedulerCallback):
    """The callback that prints a table of current progress."""

    def pre_tune(self, task_scheduler, task_id):
        if task_scheduler.tune_option.verbose < 1:
            return

        _ffi_api.PrintTitle("Task Scheduler")
        print(
            "|  ID  "
            "|                       Task Description                        "
            "| Latency (ms) | Speed (GFLOPS) | Trials |"
        )
        print(
            "----------------------------------------------------------------"
            "-------------------------------------------------"
        )

        # content
        for i in range(len(task_scheduler.tasks)):
            id_str = "%d" % i
            latency_str = (
                "%.3f" % (1e3 * task_scheduler.best_costs[i])
                if task_scheduler.best_costs[i] < 1e9
                else "-"
            )
            task_desc = task_scheduler.tasks[i].desc
            speed_str = (
                "%.2f"
                % (task_scheduler.tasks[i].compute_dag.flop_ct / task_scheduler.best_costs[i] / 1e9)
                if task_scheduler.best_costs[i] < 1e9
                else "-"
            )
            trials_str = "%d" % (task_scheduler.task_cts[i] * task_scheduler.num_measures_per_round)
            print(
                "| %4s | %61s | %12s | % 14s | %6s |"
                % (id_str, task_desc, latency_str, speed_str, trials_str)
            )
        print(
            "----------------------------------------------------------------"
            "-------------------------------------------------"
        )

        # overall info
        if all(cost < 1e9 for cost in task_scheduler.best_costs):
            total_latency_str = "%.3f" % (task_scheduler.cur_score * 1e3)
        else:
            total_latency_str = "-"
        print(
            "Estimated total latency: %s ms\tTrials: %d\tUsed time : %.0f s\tNext ID: %d\t"
            % (
                total_latency_str,
                task_scheduler.ct,
                time.time() - task_scheduler.tic,
                task_id,
            )
        )


class LogEstimatedLatency(TaskSchedulerCallback):
    """Log the estimated latency to the file after tuning a task.

    Parameters
    ----------
    log_file: str
        The log file path.
    """

    def __init__(self, log_file):
        if os.path.exists(log_file):  # Remove existing log
            os.remove(log_file)

        self.log_file = log_file

    def post_tune(self, task_scheduler, task_id):
        if all(cost < 1e9 for cost in task_scheduler.best_costs):
            total_latency_str = "%.3f" % (task_scheduler.cur_score * 1e3)
        else:
            total_latency_str = "N/A"

        with open(self.log_file, "a") as filep:
            filep.write(
                "ElapsedTime(s)\t%.0f\tEstimatedLatency(ms)\t%s\tTrials\t%d\n"
                % (
                    time.time() - task_scheduler.tic,
                    total_latency_str,
                    task_scheduler.ct,
                )
            )
            filep.flush()
