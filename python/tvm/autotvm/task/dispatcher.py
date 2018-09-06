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
from decorator import decorate

from tvm import target as _target

from .space import FallbackConfigEntity

logger = logging.getLogger('autotvm')

class DispatchContext(object):
    """
    Base class of dispatch context.

    DispatchContext enables the target and workload
    specific dispatch mechanism for templates.
    """
    current = None

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
        workload = func(*args, **kwargs)
        cfg = DispatchContext.current.query(tgt, workload)
        if cfg.is_fallback and not cfg.template_key:
            # first try 'direct' template
            if 'direct' in dispatch_dict:
                return dispatch_dict['direct'](cfg, *args, **kwargs)
            # otherwise pick a random template
            for v in dispatch_dict.values():
                return v(cfg, *args, **kwargs)
        else:
            return dispatch_dict[cfg.template_key](cfg, *args, **kwargs)

    fdecorate = decorate(fworkload, dispatch_func)
    fdecorate.register = register
    return fdecorate


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
    """
    def __init__(self, records):
        super(ApplyHistoryBest, self).__init__()

        self.best_by_targetkey = {}
        self.best_by_model = {}

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

        logger.debug("Finish loading %d records", counter)

    def _query_inside(self, target, workload):
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

        return None


class FallbackContext(DispatchContext):
    """
    A fallback dispatch context.

    Any tunable template can be called under this context.
    This is the root context.
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
