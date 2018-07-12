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

from decorator import decorate

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
