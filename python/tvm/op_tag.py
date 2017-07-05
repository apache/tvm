"""Tag class for TVM operators."""
from functools import wraps

class OpTag(object):
    """Tag scope object to set tag for operators, working as context
    manager and decorator both.

    Parameters
    ----------
    kwargs
        Keyword arguments of configurations to set.
    """
    current = None
    defaults = {
        'tag': ''
    }
    def __init__(self, **kwargs):
        self._old_scope = None
        for k, _ in kwargs.items():
            if k not in OpTag.defaults:
                raise ValueError(
                    "invalid argument %s, candidates are %s" % (k, OpTag.defaults.keys()))
        self._attr = kwargs

    def __getattr__(self, name):
        if name not in self._attr:
            return OpTag.defaults[name]
        return self._attr[name]

    def __enter__(self):
        if OpTag.current is not None:
            raise ValueError("nested op_tag is not allowed for now")
        self._old_scope = OpTag.current
        OpTag.current = self
        return self

    def __exit__(self, ptype, value, trace):
        assert self._old_scope is None
        OpTag.current = self._old_scope

    def __call__(self, fdecl):
        @wraps(fdecl)
        def tagged_fdecl(*args, **kwargs):
            with self:
                return fdecl(*args, **kwargs)
        return tagged_fdecl


def op_tag(**kwargs):
    """Configure the operator tag by setting config variables.

    Parameters
    ----------
    tag: str, default=""
        The tag name.

    Returns
    -------
    op_tag: OpTag
        The tag object
    """
    return OpTag(**kwargs)
