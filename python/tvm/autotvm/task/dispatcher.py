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
Template dispatcher module.

A dispatcher is a function that can contains multiple behaviors.
Its specific behavior is can be controlled by DispatchContext.

DispatchContext is used in two ways, usually via different implementation
of the DispatchContext base class.

- During search, we can use it to pass the current proposal from tuner.
- During evaluation, we can use it to set pick the best policy.
"""
# pylint: disable=invalid-name

from __future__ import absolute_import as _abs

import logging

import numpy as np

from .space import FallbackConfigEntity
from .. import env as _env

logger = logging.getLogger("autotvm")


class DispatchContext(object):
    """
    Base class of dispatch context.

    DispatchContext enables the target and workload
    specific dispatch mechanism for templates.
    """

    current = None
    # a set to prevent print duplicated message
    warning_messages = set()

    def __init__(self):
        self._old_ctx = DispatchContext.current

    def query(self, target, workload):
        """
        Query the context to get the specific config for a template.
        If cannot find the result inside this context, this function will query it
        from the upper contexts.

        Parameters
        ----------
        target: Target
            The current target
        workload : Workload
            The current workload.

        Returns
        -------
        cfg : ConfigSpace
            The specific configuration.
        """
        ret = self._query_inside(target, workload)
        if ret is None:
            ret = self._old_ctx.query(target, workload)
        return ret

    def update(self, target, workload, cfg):
        """
        Update context with a specific config.

        Parameters
        ----------
        target: Target
            The current target
        workload : Workload
            The current workload.
        cfg : ConfigSpace
            The specific configuration.

        Note
        ----
        This interface is for cases when TVM decides to replace an operator in the graph.
        For example, `AlterOpLayout` pass (enables when `opt_level = 3`) replaces `NCHW`
        convolution with `NCHW[x]c` implementation on x86 CPUs.
        Thus in TOPI, we first query schedule using original `NCHW` workload,
        then update the dispatcher with the new `NCHW[x]c` workload.
        So that later on, `NCHW[x]c` convolution can get schedule from the dispatcher using
        its own workload directly.

        .. code-block:: python

            @conv2d_alter_layout.register("cpu")
            def _alter_conv2d_layout(attrs, inputs, tinfo):
                workload = get_conv2d_workload(...)
                dispatch_ctx = autotvm.task.DispatchContext.current
                target = tvm.target.Target.current()
                config = dispatch_ctx.query(target, workload)

                # Get conv2d_NCHWc workload from config
                # new_workload = ...
                # new_inputs = ...
                # new_attrs = ...

                # Store altered operator's config
                dispatch_ctx.update(target, new_workload, config)
                return sym.contrib.conv2d_NCHWc(*new_inputs, **new_attrs)

        We directly store `config` back because `conv2d_NCHW` and `conv2d_NCHWc`
        share the same schedule parameters.
        One can construct a new `ConfigEntity` if this is not the case.
        """
        raise NotImplementedError()

    def _query_inside(self, target, workload):
        """
        Query the context to get the specific config for a template.
        This function only query config inside this context.

        Parameters
        ----------
        target: Target
            The current target
        workload : Workload
            The current workload.

        Returns
        -------
        cfg : ConfigSpace
            The specific configuration.
        """
        raise NotImplementedError()

    def __enter__(self):
        self._old_ctx = DispatchContext.current
        DispatchContext.current = self
        return self

    def __exit__(self, ptype, value, trace):
        DispatchContext.current = self._old_ctx


class ApplyConfig(DispatchContext):
    """Apply a deterministic config entity for all queries.

    Parameters
    ----------
    config : ConfigSpace or ConfigEntity
        The specific configuration we care about.
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
    records : str or iterator of (autotvm.measure.MeasureInput, autotvm.measure.MeasureResult)
        Collection of tuning records.
        If is str, then it should be the filename of a records log file.
        Each row of this file is an encoded record pair. Otherwise, it is an iterator.
    """

    def __init__(self, records):
        super(ApplyHistoryBest, self).__init__()

        self.best_by_targetkey = {}
        self.best_by_model = {}
        self._best_user_defined = {}

        if records:
            self.load(records)

    def load(self, records):
        """Load records to this dispatch context

        Parameters
        ----------
        records : str or iterator of (autotvm.measure.MeasureInput, autotvm.measure.MeasureResult)
            Collection of tuning records.
            If is str, then it should be the filename of a records log file.
            Each row of this file is an encoded record pair. Otherwise, it is an iterator.
        """
        # pylint: disable=import-outside-toplevel
        from pathlib import Path
        from ..record import load_from_file

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
            counter += 1
            if res.error_no != 0:
                continue

            # use target keys in tvm target system as key to build best map
            for k in inp.target.keys:
                key = (k, inp.task.workload)
                if key not in best_by_targetkey:
                    best_by_targetkey[key] = (inp, res)
                else:
                    _, other_res = best_by_targetkey[key]
                    if np.mean(other_res.costs) > np.mean(res.costs):
                        best_by_targetkey[key] = (inp, res)

            # use model as key to build best map
            key = (inp.target.model, inp.task.workload)
            if key not in best_by_model:
                if inp.target.model != "unknown":
                    best_by_model[key] = (inp, res)
            else:
                _, other_res = best_by_model[key]
                if np.mean(other_res.costs) > np.mean(res.costs):
                    best_by_model[key] = (inp, res)

        logger.debug("Finish loading %d records", counter)

    def _query_inside(self, target, workload):
        if target is None:
            raise RuntimeError(
                "Need a target context to find the history best. "
                "Hint: If your target is llvm, use `with tvm.target.Target('llvm'):`"
                " above the dispatcher call. So does other target. "
            )

        # first try matching by model
        key = (target.model, workload)
        if key in self._best_user_defined:
            return self._best_user_defined[key]
        if key in self.best_by_model:
            inp, _ = self.best_by_model[key]
            return inp.config

        # then try matching by target key
        for k in target.keys:
            key = (k, workload)
            if key in self._best_user_defined:
                return self._best_user_defined[key]
            if key in self.best_by_targetkey:
                inp, _ = self.best_by_targetkey[key]
                return inp.config

        return None

    def update(self, target, workload, cfg):
        model = target.model
        key = (model, workload)
        # assume user provided config is the best
        cfg.cost = 0
        self._best_user_defined[key] = cfg

        for k in target.keys:
            key = (k, workload)
            self._best_user_defined[key] = cfg


class FallbackContext(DispatchContext):
    """
    A fallback dispatch context.

    Any tunable template can be called under this context.
    This is the root context.
    """

    def __init__(self):
        super(FallbackContext, self).__init__()
        self.memory = {}

    def _query_inside(self, target, workload):
        key = (str(target), workload)
        if key in self.memory:
            return self.memory[key]

        if not _env.GLOBAL_SCOPE.silent:
            msg = (
                "Cannot find config for target=%s, workload=%s. A fallback configuration "
                "is used, which may bring great performance regression." % (target, workload)
            )
            if msg not in DispatchContext.warning_messages:
                DispatchContext.warning_messages.add(msg)
                logger.warning(msg)
        cfg = FallbackConfigEntity()

        # cache this config
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


def clear_fallback_cache(target, workload):
    """Clear fallback cache. Pass the same argument as _query_inside to this function
    to clean the cache.

    Parameters
    ----------
    target: Target
        The current target
    workload : Workload
        The current workload.

    Note
    ----
    This is used in alter_op_layout to clear the bad cache created before call topi compute function
    """
    context = DispatchContext.current
    while not isinstance(context, FallbackContext):
        context = context._old_ctx
    context.clear_cache(target, workload)


class ApplyGraphBest(DispatchContext):
    """Load the graph level tuning optimal schedules.

    The input records should be in the ascending order of
    node index for target operator. Usually this can be obtained
    with graph tuner.

    This context maintains an internal counter to indicate the current
    node index.
    """

    def __init__(self, records):
        """
        Parameters
        ----------
        records : str or iterator of (autotvm.measure.MeasureInput, autotvm.measure.MeasureResult)
            Collection of tuning records.
            If is str, then it should be the filename of a records log file.
                   Each row of this file is an encoded record pair.
            Otherwise, it is an iterator.
        """
        # pylint: disable=import-outside-toplevel
        from ..record import load_from_file

        super(ApplyGraphBest, self).__init__()
        if isinstance(records, str):
            records = load_from_file(records)
        self._records = list(records)
        self._counter = 0
        self._global_cfg_dict = {}

    def _query_inside(self, target, workload):
        """
        Query the context to get config from records.

        Parameters
        ----------
        target : Target
            The current target
        workload : Workload
            The current workload.

        Returns
        -------
        cfg : ConfigSpace
            The specific configuration.
        """
        if self._counter < len(self._records):
            cfg = self._records[self._counter][0].config
            wkl = self._records[self._counter][0].task.workload
            if workload is not None:
                assert wkl == workload
            self._counter += 1
            self.update(target, wkl, cfg)
            cfg.workload = wkl
            return cfg
        key = (str(target), workload)
        if key not in self._global_cfg_dict:
            msg = (
                "Config for target=%s, workload=%s is missing in ApplyGraphBest context. "
                "A fallback configuration is used, which may bring great performance "
                "regression." % (target, workload)
            )
            logger.warning(msg)
            cfg = FallbackConfigEntity()
            self._global_cfg_dict[key] = cfg
        else:
            cfg = self._global_cfg_dict[key]
        return cfg

    def update(self, target, workload, cfg):
        key = (str(target), workload)
        self._global_cfg_dict[key] = cfg
