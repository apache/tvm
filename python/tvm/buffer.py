from __future__ import absolute_import as _abs
from . import expr as _expr
from . import expr_util as _expr_util
from . import var_name as _name


def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)


"""Scope defines the scope of a buffer

Types
-----
Thread :  thread private buffer (registers)
Shared :  shared buffer within a thread block (shared memory)
Global :  buffer in the global GPU RAM
"""
Scope = enum('Thread', 'Shared', 'Global')


class Buffer(object):
    def __init__(self, scope, name=None):
        self.scope = scope
        buf_name = 'Buffer_'
        if name: buf_name += name
        self.name = _name.NameManager.current.get(buf_name)
        self.shape = []
        self.offset_index = []

    def reshape(self, domain):
        for r in domain:
            self.shape.append(r.extent)
            self.offset_index.append(r.begin)

    def __call__(self, *global_index):
        if len(global_index) != len(self.shape):
            raise ValueError("Need to provide %d index in buffer slice" % len(self.shape))
        stride = [1]
        for i in reversed(range(1, len(self.shape))):
            stride.insert(0, self.shape[i] * stride[0])
        local_index = []
        for i in range(0, len(global_index)):
            local_index.append(global_index[i] - self.offset_index[i])
        index = local_index[0] * stride[0]
        for i in range(1, len(local_index)):
            index = index + local_index[i] * stride[i]
        index = _expr_util.simplify(index)
        return _expr.TensorRefExpr(self, [index])


class BufferManager(object):
    def __init__(self):
        self._buffer_map = {}
        self._old_manager = None

    def get(self, tensor):
        if tensor in self._buffer_map:
            return self._buffer_map[tensor]
        return None

    def bind(self, tensor, buf):
        self._buffer_map[tensor] = buf

    def __enter__(self):
        self._old_manager = BufferManager.current
        BufferManager.current = self
        return self

    def __exit__(self, ptype, value, trace):
        assert self._old_manager
        BufferManager.current = self._old_manager

# initialize the default buffer manager
BufferManager.current = BufferManager()
