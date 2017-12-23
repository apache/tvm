"""Tag class for TVM operators."""
from ._ffi.base import _LIB_NAME
try:
    from decorator import decorate
except ImportError as err_msg:
    # Allow decorator to be missing in runtime
    if _LIB_NAME != "libtvm_runtime.so":
        raise err_msg

class TagScope(object):
    """Tag scope object to set tag for operators, working as context
    manager and decorator both. See also tag_scope.
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
        def tagged_fdecl(func, *args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return decorate(fdecl, tagged_fdecl)


def tag_scope(tag):
    """The operator tag scope.

    Parameters
    ----------
    tag: str
        The tag name.

    Returns
    -------
    tag_scope: TagScope
        The tag scope object, which can be used as decorator or
        context manger.

    Example
    -------
    .. code-block:: python

        n = tvm.var('n')
        m = tvm.var('m')
        l = tvm.var('l')
        A = tvm.placeholder((n, l), name='A')
        B = tvm.placeholder((m, l), name='B')
        k = tvm.reduce_axis((0, l), name='k')

        with tvm.tag_scope(tag='matmul'):
            C = tvm.compute((n, m), lambda i, j: tvm.sum(A[i, k] * B[j, k], axis=k))

        # or use tag_scope as decorator
        @tvm.tag_scope(tag="conv")
        def compute_relu(data):
            return tvm.compute(data.shape, lambda *i: tvm.select(data(*i) < 0, 0.0, data(*i)))
    """
    return TagScope(tag)
