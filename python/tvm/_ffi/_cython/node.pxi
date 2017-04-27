from ..base import string_types
from ..node_generic import _set_class_node_base

"""Maps node type to its constructor"""
NODE_TYPE = []

def _register_node(int index, object cls):
    """register node class"""
    while len(NODE_TYPE) <= index:
        NODE_TYPE.append(None)
    NODE_TYPE[index] = cls

cdef inline object make_ret_node(void* chandle):
    global NODE_TYPE
    cdef int tindex
    cdef list node_type
    cdef object cls
    node_type = NODE_TYPE
    CALL(TVMNodeGetTypeIndex(chandle, &tindex))
    if tindex < len(node_type):
        cls = node_type[tindex]
        if cls is not None:
            obj = cls(None)
        else:
            obj = NodeBase(None)
    (<NodeBase>obj).chandle = chandle
    return obj

cdef class NodeBase:
    cdef void* chandle

    cdef _set_handle(self, handle):
        cdef unsigned long long ptr
        if handle is None:
            self.chandle = NULL
        else:
            ptr = handle.value
            self.chandle = <void*>(ptr)

    property handle:
        def __get__(self):
            if self.chandle == NULL:
                return None
            else:
                return ctypes_handle(self.chandle)

        def __set__(self, value):
            self._set_handle(value)

    def __init__(self, handle):
        self._set_handle(handle)

    def __dealloc__(self):
        CALL(TVMNodeFree(self.chandle))

    def __getattr__(self, name):
        cdef TVMValue ret_val
        cdef int ret_type_code, ret_succ
        CALL(TVMNodeGetAttr(self.chandle, c_str(name),
                            &ret_val, &ret_type_code, &ret_succ))
        if ret_succ == 0:
            raise AttributeError(
                "'%s' object has no attribute '%s'" % (type(self), name))
        return make_ret(ret_val, ret_type_code)

_set_class_node_base(NodeBase)
