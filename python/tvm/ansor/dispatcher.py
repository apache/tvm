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
The global context that dispatches best configurations to workloads
"""
# pylint: disable=invalid-name

from __future__ import absolute_import as _abs

import logging

import numpy as np

from tvm.tir.expr import FloatImm

logger = logging.getLogger('auto_scheduler')


class DispatchContext(object):
    """
    Base class of dispatch context.
    """
    current = None

    def __init__(self):
        self._old_ctx = DispatchContext.current

    def query(self, target, workload):
        """
        Query the context to get the specific config for a workload.
        If cannot find the result inside this context, this function will query it
        from the upper contexts.

        Parameters
        ----------
        target: Target
            The current target
        workload : str
            The current workload

        Returns
        -------
        cfg : State
            The schedule configuration for the workload
        """
        ret = self._query_inside(target, workload)
        return ret

    def update(self, target, workload, cfg):
        """
        Update the config for a workload

        Parameters
        ----------
        target: Target
            The current target
        workload : Workload
            The current workload.
        cfg : State
            The schedule configuration for the workload
        """
        raise NotImplementedError()

    def _query_inside(self, target, workload):
        """
        Query the context to get the specific config for a workload.
        This function only query config inside this context.

        Parameters
        ----------
        target: Target
            The current target
        workload : Workload
            The current workload.

        Returns
        -------
        cfg : State or str
            The schedule configuration for the workload
        """
        raise NotImplementedError()

    def __enter__(self):
        self._old_ctx = DispatchContext.current
        DispatchContext.current = self
        return self

    def __exit__(self, ptype, value, trace):
        DispatchContext.current = self._old_ctx


class ApplyConfig(DispatchContext):
    """Apply a deterministic config for all queries.

    Parameters
    ----------
    config : State
        The schedule configuration
    """
    def __init__(self, config):
        super(ApplyConfig, self).__init__()
        self._config = config
        self.workload = None

    def _query_inside(self, target, workload):
        """Override query"""
        self.workload = workload
        return self._config

    def update(self, target, workload, cfg):
        """Override update"""
        self.workload = workload
        self._config = cfg


class ApplyHistoryBest(DispatchContext):
    """
    Apply the history best config

    Parameters
    ----------
    records : str or iterator of (MeasureInput, MeasureResult)
        Collection of tuning records.
        If is str, then it should be the filename of a records log file.
                   Each row of this file is an encoded record pair.
        Otherwise, it is an iterator.
    n_lines: int (optional)
        if it is not None, only load the first `n_lines` lines of log
    """
    def __init__(self, records, n_lines=None):
        super(ApplyHistoryBest, self).__init__()

        self.best_by_targetkey = {}
        self.best_by_model = {}
        self._best_user_defined = {}

        if records:
            self.load(records, n_lines)

    def load(self, records, n_lines=None):
        """Load records to this dispatch context

        Parameters
        ----------
        records : str or iterator of (MeasureInput, MeasureResult)
            Collection of tuning records.
            If is str, then it should be the filename of a records log file.
                       Each row of this file is an encoded record pair.
            Otherwise, it is an iterator.
        n_lines: int (optional)
            if it is not None, only load the first `n_lines` lines of log
        """
        from pathlib import Path
        from . import load_from_file

        if isinstance(records, Path):
            records = str(records)

        if isinstance(records, str):
            records = load_from_file(records)
        if not records:
            return

        best_by_targetkey = self.best_by_targetkey
        best_by_model = self.best_by_model

        counter = 0
        for inp, res in records:
            if n_lines is not None and counter >= n_lines:
                break
            counter += 1
            if res.error_no != 0:
                continue

            # use target keys in tvm target system as key to build best map
            for k in inp.task.target.keys:
                key = (k, inp.task.workload_key)
                if key not in best_by_targetkey:
                    best_by_targetkey[key] = (inp, res)
                else:
                    _, other_res = best_by_targetkey[key]
                    other_costs = [x.value for x in other_res.costs if isinstance(x, FloatImm)]
                    costs = [x.value for x in res.costs if isinstance(x, FloatImm)]
                    if np.mean(other_costs) > np.mean(costs):
                        best_by_targetkey[key] = (inp, res)

            # use model as key to build best map
            key = (inp.task.target.model, inp.task.workload_key)
            if key not in best_by_model:
                if inp.task.target.model != 'unknown':
                    best_by_model[key] = (inp, res)
            else:
                _, other_res = best_by_model[key]
                other_costs = [x.value for x in other_res.costs if isinstance(x, FloatImm)]
                costs = [x.value for x in res.costs if isinstance(x, FloatImm)]
                if np.mean(other_costs) > np.mean(costs):
                    best_by_model[key] = (inp, res)

        logger.debug("Finish loading %d records", counter)

    def _query_inside(self, target, workload):
        if target is None:
            raise RuntimeError("Need a target context to find the history best. "
                               "Hint: If your target is llvm, use `with tvm.target.create('llvm'):`"
                               " above the dispatcher call. So does other target. ")

        # first try matching by model
        key = (target.model, workload)
        if key in self._best_user_defined:
            return self._best_user_defined[key]
        if key in self.best_by_model:
            return self.best_by_model[key][0].state

        # then try matching by target key
        for k in target.keys:
            key = (k, workload)
            if key in self._best_user_defined:
                return self._best_user_defined[key]
            if key in self.best_by_targetkey:
                return self.best_by_targetkey[key][0].state

        return None

    def update(self, target, workload, state):
        model = target.model
        key = (model, workload)
        self._best_user_defined[key] = state

        for k in target.keys:
            key = (k, workload)
            self._best_user_defined[key] = state


class FallbackContext(DispatchContext):
    """
    A fallback dispatch context.
    This is used as the root context.
    """

    def __init__(self):
        super(FallbackContext, self).__init__()
        self.memory = {}
        self.silent = False

        # a set to prevent print duplicated message
        self.messages = set()

    def _query_inside(self, target, workload):
        key = (str(target), workload)
        if key in self.memory:
            return self.memory[key]

        if not self.silent:
            msg = "Cannot find config for target=%s, workload=%s. A fallback configuration "\
                  "is used, which may bring great performance regression." % (target, workload)
            if msg not in self.messages:
                self.messages.add(msg)
                logger.warning(msg)
        cfg = None

        # cache this config to avoid duplicated warning message
        self.memory[key] = cfg
        return cfg

    def clear_cache(self, target, workload):
        """Clear fallback cache. Pass the same argument as _query_inside to this function
        to clean the cache.

        Parameters
        ----------
        target: Target
            The current target
        workload : Workload
            The current workload.
        """
        key = (str(target), workload)
        if key in self.memory:
            del self.memory[key]

    def update(self, target, workload, cfg):
        key = (str(target), workload)
        self.memory[key] = cfg


DispatchContext.current = FallbackContext()
