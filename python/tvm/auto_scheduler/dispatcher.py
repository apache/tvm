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
The global context that dispatches best schedules to workloads.

In auto-scheduler, a state (loop_state.py::StateObject) saves the
schedule configuration by its transform_steps, so a state is used
as a schedule configuration here.
"""
# pylint: disable=invalid-name

import logging
import pathlib
from collections.abc import Iterable

import numpy as np

from tvm.contrib.utils import tempdir
from tvm.tir.expr import FloatImm
from .cost_model import RandomModel, XGBModel
from .measure import LocalRPCMeasureContext
from .measure_record import RecordToFile, load_records
from .search_policy import PreloadMeasuredStates, SketchPolicy
from .search_task import SearchTask, TuningOptions
from .utils import calc_workload_dis_factor, decode_workload_key

logger = logging.getLogger("auto_scheduler")


class DispatchContext(object):
    """
    Base class of dispatch context.
    """

    current = None

    def __init__(self):
        self._old_ctx = DispatchContext.current

    def query(self, target, workload_key, has_complex_op, dag, func_name):
        """
        Query the context to get the specific config for a workload.
        If this function cannot find the result inside this context, it will query the result
        from the upper contexts.

        Parameters
        ----------
        target: Target
            The current target
        workload_key : str
            The workload key
        has_complex_op: bool
            Whether this workload has at least one complex op.
        dag: ComputeDAG
            The ComputeDAG of the workload.
        func_name: str
            The function name of this workload.

        Returns
        -------
        state : StateObject
            The state that stores schedule configuration for the workload
        """
        ret = self._query_inside(target, workload_key, func_name)
        if ret is None:
            ret = self._old_ctx.query(target, workload_key, has_complex_op, dag, func_name)
        return ret

    def update(self, target, workload_key, state):
        """
        Update the config for a workload

        Parameters
        ----------
        target: Target
            The current target
        workload_key : str
            The current workload_key.
        state : StateObject
            The state that stores schedule configuration for the workload
        """
        raise NotImplementedError()

    def _query_inside(self, target, workload_key, func_name):
        """
        Query the context to get the specific config for a workload.
        This function only query config inside this context.

        Parameters
        ----------
        target: Target
            The current target
        workload_key : str
            The current workload_key.
        func_name: str
            The function name of this workload.

        Returns
        -------
        state : StateObject
            The schedule configuration for the workload
        """
        raise NotImplementedError()

    def __enter__(self):
        self._old_ctx = DispatchContext.current
        DispatchContext.current = self
        return self

    def __exit__(self, ptype, value, trace):
        DispatchContext.current = self._old_ctx


class ApplyHistoryBest(DispatchContext):
    """
    Apply the history best config

    Parameters
    ----------
    records : str, list of str, or iterator of (auto_scheduler.measure.MeasureInput,\
                                                auto_scheduler.measure.MeasureResult)
        Collection of tuning records.
        If is str, then it should be the filename of a records log file.
        Each row of this file is an encoded record pair. If it is an iterator,
        it can either be a set of str filenames which will be applied jointly,
        or a set of (input, result) tuples.
    n_lines: Optional[int]
        if it is not None, only load the first `n_lines` lines of log.
    include_compatible: bool
        When set to True, compatible records will also be considered.
    """

    def __init__(self, records, n_lines=None, include_compatible=False):
        super(ApplyHistoryBest, self).__init__()
        self.include_compatible = include_compatible

        # Dict[str (target key),
        #   Dict[str (workload hash),
        #     Dict[tuple (workload args), tuple (State, cost)]]]
        self.best_by_targetkey = {}
        self.best_by_model = {}
        self._best_user_defined = {}

        self.load(records, n_lines)

    @staticmethod
    def get_workload_entry(best_records, target_key, workload_key):
        """Get the entry of the target key and workload key hash in the given best record map.

        Parameters
        ----------
        best_records: Dict[str, Dict[str, Dict[str, Any]]]
            The best record map.
        target_key: str
            The first key to the best_records.
        workload_key: str
            The workload key that can be decoded to workload hash and args.

        Returns
        -------
        entry: Dict[str, Any]
            The entry in best_records with target key and workload hash.
        workload_hash: str
            The workload hash decoded from workload_key.
        workload_args: Tuple[Any, ...]
            The hashable tuple of workload args decoded from workload_key.
        """
        workload_hash, workload_args = decode_workload_key(workload_key)
        if target_key not in best_records:
            best_records[target_key] = {}
        if workload_hash not in best_records[target_key]:
            best_records[target_key][workload_hash] = {}
        return best_records[target_key][workload_hash], workload_hash, workload_args

    def load(self, records, n_lines=None):
        """Load records to this dispatch context

        Parameters
        ----------
        records : str or iterator of (auto_scheduler.measure.MeasureInput,\
                                      auto_scheduler.measure.MeasureResult)
            Collection of tuning records.
            If is str, then it should be the filename of a records log file.
            Each row of this file is an encoded record pair. Otherwise, it is an iterator.
        n_lines: Optional[int]
            if it is not None, only load the first `n_lines` lines of log
        """
        joint_records = []
        if not isinstance(records, Iterable) or isinstance(records, str):
            records = [records]

        for rec in records:
            if isinstance(rec, pathlib.Path):
                rec = str(rec)

            if isinstance(rec, str):
                rec = load_records(rec)
                joint_records += rec
            else:
                if rec is not None:
                    joint_records.append(rec)

        if not joint_records:
            return

        best_by_targetkey = self.best_by_targetkey
        best_by_model = self.best_by_model

        counter = 0
        for inp, res in joint_records:
            if n_lines is not None and counter >= n_lines:
                break
            counter += 1
            if res.error_no != 0:
                continue

            costs = [x.value for x in res.costs if isinstance(x, FloatImm)]
            cost = np.mean(costs)

            # use target keys in tvm target system as key to build best map
            for k in inp.task.target.keys:
                entry, _, workload_args = self.get_workload_entry(
                    best_by_targetkey, k, inp.task.workload_key
                )
                if workload_args not in entry:
                    entry[workload_args] = (inp.state, cost)
                else:
                    _, other_cost = entry[workload_args]
                    if other_cost > cost:
                        entry[workload_args] = (inp.state, cost)

            # use model as key to build best map
            entry, _, workload_args = self.get_workload_entry(
                best_by_model, inp.task.target.model, inp.task.workload_key
            )
            if workload_args not in entry:
                if inp.task.target.model != "unknown":
                    entry[workload_args] = (inp.state, cost)
            else:
                _, other_cost = entry[workload_args]
                if other_cost > cost:
                    entry[workload_args] = (inp.state, cost)

        logger.debug("Finish loading %d records", counter)

    def _query_inside(self, target, workload_key, func_name):
        if target is None:
            raise RuntimeError(
                "Need a target context to find the history best. "
                "Hint: If your target is llvm, use `with tvm.target.create('llvm'):`"
                " above the dispatcher call. So does other target. "
            )

        def match_record(best_records, target_key, workload_key):
            """The helper function to match the record in the given map
            and return the matched state, or None if no match.
            """
            ret = None

            entry, workload_hash, workload_args = self.get_workload_entry(
                best_records, target_key, workload_key
            )
            if workload_args in entry:
                ret = entry[workload_args][0]
            elif self.include_compatible:
                best_cost = float("inf")
                for args, val in entry.items():
                    dis_f = calc_workload_dis_factor(
                        (workload_hash, workload_args), (workload_hash, args)
                    )
                    if dis_f == float("inf"):
                        continue

                    state, cost = val
                    cost *= dis_f
                    if ret is None or cost < best_cost:
                        best_cost = cost
                        ret = state
            return ret

        # first try matching by model
        ret = match_record(self._best_user_defined, target.model, workload_key)
        if ret is not None:
            return ret
        ret = match_record(self.best_by_model, target.model, workload_key)
        if ret is not None:
            return ret

        # then try matching by target key
        for k in target.keys:
            ret = match_record(self._best_user_defined, k, workload_key)
            if ret is not None:
                return ret
            ret = match_record(self.best_by_targetkey, k, workload_key)
            if ret is not None:
                return ret

        return None

    def update(self, target, workload_key, state):
        entry, _, workload_args = self.get_workload_entry(
            self._best_user_defined, target.model, workload_key
        )
        entry[workload_args] = (state, 1)

        for k in target.keys:
            entry, _, _ = self.get_workload_entry(self._best_user_defined, k, workload_key)
            entry[workload_args] = (state, 1)


class ApplyHistoryBestOrSample(ApplyHistoryBest):
    """
    Apply the history best config, or sample a valid schedule if no config is found.

    Parameters
    ----------
    records : str or iterator of (auto_scheduler.measure.MeasureInput,\
                                  auto_scheduler.measure.MeasureResult)
        Collection of tuning records.
        If is str, then it should be the filename of a records log file.
        Each row of this file is an encoded record pair. Otherwise, it is an iterator.
    sample_simple_workloads: bool
        When False, sampling will not apply to simple workloads (w/o reduction).
    cost_model_file: str
        The filename of the pre-trained XGBoost cost model. If not present, then random
        model will be used.
    num_measure: int
        Meausre the top-N rank of sampled schedules on the device. The default -1 means
        no measurement and simply return the top-1 schedule ranked by the cost model.
    """

    def __init__(
        self, records, sample_simple_workloads=False, cost_model_file=None, num_measure=-1
    ):
        self.sample_simple_workloads = sample_simple_workloads
        self.num_measure = num_measure
        self.log_dir = tempdir()
        if cost_model_file is None:
            self.cost_model = RandomModel()
        else:
            self.cost_model = XGBModel()
            self.cost_model.load(cost_model_file)

        super(ApplyHistoryBestOrSample, self).__init__(
            records, n_lines=None, include_compatible=True
        )

    def query(self, target, workload_key, has_complex_op, dag, func_name):
        if has_complex_op or self.sample_simple_workloads:
            ret = self._query_inside(target, workload_key, func_name)
        else:
            ret = super(ApplyHistoryBestOrSample, self)._query_inside(
                target, workload_key, func_name
            )

        if ret is None:
            ret = self._old_ctx.query(target, workload_key, has_complex_op, dag, func_name)
        return ret

    def _query_inside(self, target, workload_key, func_name):
        ret = super(ApplyHistoryBestOrSample, self)._query_inside(target, workload_key, func_name)
        if ret is not None:
            return ret

        # Sampling valid schedules when no existing records can be used.
        task = SearchTask(workload_key=workload_key, target=target)
        measure_ctx = LocalRPCMeasureContext(min_repeat_ms=300)

        log_file = self.log_dir.relpath(f"{decode_workload_key(workload_key)[0]}.log")

        while ret is None:
            tune_option = TuningOptions(
                num_measure_trials=self.num_measure,
                runner=measure_ctx.runner,
                measure_callbacks=[RecordToFile(log_file)],
                verbose=0,
            )
            search_policy = SketchPolicy(
                task,
                self.cost_model,
                params={
                    "eps_greedy": 0.01,
                    "sample_init_min_population": 64,
                    "evolutionary_search_num_iters": 0,
                },
                init_search_callbacks=[PreloadMeasuredStates(log_file)],
                verbose=0,
            )
            task.tune(tune_option, search_policy)

            # Load the sampled records and query again.
            self.load(log_file)
            ret = super(ApplyHistoryBestOrSample, self)._query_inside(
                target, workload_key, func_name
            )

        del measure_ctx
        return ret


class FallbackContext(DispatchContext):
    """
    A fallback dispatch context.
    This is used as the root context.
    """

    def __init__(self):
        super(FallbackContext, self).__init__()
        self.memory = {}

        # Verbose level:
        # 0: Completely silent.
        # 1: Warning the missing configs for querying complex tasks.
        # 2: Warning the missing configs for querying all tasks.
        self.verbose = 1

        # a set to prevent print duplicated message
        self.messages = set()

    def query(self, target, workload_key, has_complex_op, dag, func_name):
        key = (str(target), workload_key)
        if key in self.memory:
            return self.memory[key]

        if self.verbose == 2 or (has_complex_op and self.verbose == 1):
            msg = (
                f"-----------------------------------\n"
                f"{func_name}\n"
                f"Cannot find tuned schedules for target={target}, workload_key={workload_key}. "
                f"A fallback TOPI schedule is used, "
                f"which may bring great performance regression or even compilation failure. "
                f"Compute DAG info:\n{dag}"
            )
            if msg not in self.messages:
                self.messages.add(msg)
                logger.warning(msg)

        state = None

        # cache this config to avoid duplicated warning message
        self.memory[key] = state
        return state

    def _query_inside(self, target, workload_key, func_name):
        _ = target = workload_key = func_name
        raise RuntimeError("This function should never be called")

    def update(self, target, workload_key, state):
        key = (str(target), workload_key)
        self.memory[key] = state


DispatchContext.current = FallbackContext()
