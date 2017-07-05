"""Tag class for TVM operators."""
from functools import wraps

class TagScope(object):
    """Tag scope object to set tag for operators, working as context
    manager and decorator both.
    """
    current = None
    def __init__(self, tag):
        self._old_scope = None
        self.tag = tag

    def __enter__(self):
        if TagScope.current is not None:
            raise ValueError("nested op_tag is not allowed for now")
        self._old_scope = TagScope.current
        TagScope.current = self
        return self

    def __exit__(self, ptype, value, trace):
        assert self._old_scope is None
        TagScope.current = self._old_scope

    def __call__(self, fdecl):
        @wraps(fdecl)
        def tagged_fdecl(*args, **kwargs):
            with self:
                return fdecl(*args, **kwargs)
        return tagged_fdecl


def tag_scope(tag):
    """The operator tag scope.

    Parameters
    ----------
    tag: str, default=""
        The tag name.

    Returns
    -------
    tag_scope: TagScope
        The tag scope object, which can be used as decorator or
        context manger.
    """
    return TagScope(tag)
