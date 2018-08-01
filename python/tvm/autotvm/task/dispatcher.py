"""
Template dispatcher module.

A dispatcher is a function that can contains multiple behaviors.
Its specific behavior is can be controlled by DispatchContext.

DispatchContext is used in two ways, usually via different implementation
of the DispatchContext base class.

- During search, we can use it to pass the current proposal from tuner.
- During evaluation, we can use it to set pick the best policy.
"""
from __future__ import absolute_import as _abs

import logging

from decorator import decorate
import numpy as np

from tvm import target as _target

class DispatchContext(object):
    """
    Base class of dispatch context.

    DispatchContext enables the target and workload
    specific dispatch mechanism for templates.
    """
    current = None

    def query(self, target, workload):
        """
        Query the context to get the specific implementation.

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


def dispatcher(fworkload):
    """Wrap a workload dispatcher function.

    Parameters
    ----------
    fworkload : function
        The workload extraction function from arguments.

    Returns
    -------
    fdispatcher : function
        A wrapped dispatcher function, which will
        dispatch based on DispatchContext and
        the current workload.
    """
    dispatch_dict = {}
    func_name = fworkload.__name__

    def register(key, func=None, override=False):
        """Register template function.

        Parameters
        ----------
        key : str or List of str
            The template key to identify the template
            under this dispatcher.
        func : function
            The function to be registered.
            The first argument of the function is always
            cfg returned by DispatchContext,
            the rest arguments are the same as the fworkload.
        override : bool
            Whether override existing registration.

        Returns
        -------
        The register function if necessary.
        """
        if isinstance(key, str):
            key = [key]

        def _do_reg(myf):
            for x in key:
                if x in dispatch_dict and not override:
                    raise ValueError(
                        "Key %s is already registered for %s" % (x, func_name))
                dispatch_dict[x] = myf
            return myf

        if func:
            return _do_reg(func)
        return _do_reg

    def dispatch_func(func, *args, **kwargs):
        """The wrapped dispatch function"""
        tgt = _target.current_target()
        context = DispatchContext.current
        if context is None:
            raise RuntimeError("DispatchContext is not initialized")
        workload = func(*args, **kwargs)
        cfg = context.query(tgt, workload)
        return dispatch_dict[cfg.template_key](cfg, *args, **kwargs)

    fdecorate = decorate(fworkload, dispatch_func)
    fdecorate.register = register
    return fdecorate


class ApplyConfig(DispatchContext):
    """Apply a specific config entity during query.

    Parameters
    ----------
    config : ConfigSpace or ConfigEntity
        The specific configuration we care about.
    """
    def __init__(self, config):
        super(ApplyConfig, self).__init__()
        self._config = config
        self.workload = None

    def query(self, target, workload):
        """Override query"""
        self.workload = workload
        return self._config


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
    default: ConfigEntity, optional
        The default config to return when no history records
    """
    def __init__(self, records, default=None):
        super(ApplyHistoryBest, self).__init__()

        self.best_by_targetkey = {}
        self.best_by_model = {}
        self._default = default

        if records:
            self.load(records)

    def load(self, records):
        """Load records to this dispatch context

        Parameters
        ----------
        records : str or iterator of (MeasureInput, MeasureResult)
            Collection of tuning records.
            If is str, then it should be the filename of a records log file.
                       Each row of this file is an encoded record pair.
            Otherwise, it is an iterator.
        """
        from ..record import load_from_file

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
            for opt in inp.target.options:
                if opt.startswith("-model"):
                    model = opt[7:]
                    key = (model, inp.task.workload)
                    if key not in best_by_model:
                        best_by_model[key] = (inp, res)
                    else:
                        _, other_res = best_by_model[key]
                        if np.mean(other_res.costs) > np.mean(res.costs):
                            best_by_model[key] = (inp, res)
                    break

        logging.debug("Finish loading %d records", counter)

    def query(self, target, workload):
        if target is None:
            raise RuntimeError("Need a target context to find the history best. "
                               "Hint: If your target is llvm, use `with tvm.target.create('llvm'):`"
                               " above the dispatcher call. So does other target. ")

        # first try matching by model
        for opt in target.options:
            if opt.startswith("-model"):
                model = opt[7:]
                key = (model, workload)
                if key in self.best_by_model:
                    return self.best_by_model[key][0].config

        # then try matching by target key
        for k in target.keys:
            key = (k, workload)
            if key in self.best_by_targetkey:
                return self.best_by_targetkey[key][0].config

        if self._default:
            return self._default
        raise RuntimeError(
            "Cannot find config for target=%s, workload=%s" % (target, workload))
