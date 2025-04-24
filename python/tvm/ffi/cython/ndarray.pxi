cdef class NDArray:
    cdef DLTensor* _dltensor


cdef inline object make_ret_dltensor(TVMFFIAny result):
    # TODO: Implement
    return None
