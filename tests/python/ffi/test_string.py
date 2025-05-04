from tvm import ffi as tvm_ffi


def test_string():
    fecho = tvm_ffi.get_global_func("testing.echo")
    s = tvm_ffi.String("hello")
    assert isinstance(s, tvm_ffi.String)
    s2 = fecho(s)
    assert s2.__tvm_ffi_object__.same_as(s.__tvm_ffi_object__)

    s3 = tvm_ffi.convert("hello")
    assert isinstance(s3, tvm_ffi.String)
    assert isinstance(s3, str)


def test_bytes():
    fecho = tvm_ffi.get_global_func("testing.echo")
    b = tvm_ffi.Bytes(b"hello")
    assert isinstance(b, tvm_ffi.Bytes)
    b2 = fecho(b)
    assert b2.__tvm_ffi_object__.same_as(b.__tvm_ffi_object__)

    b3 = tvm_ffi.convert(b"hello")
    assert isinstance(b3, tvm_ffi.Bytes)
    assert isinstance(b3, bytes)

    b4 = tvm_ffi.convert(bytearray(b"hello"))
    assert isinstance(b4, tvm_ffi.Bytes)
    assert isinstance(b4, bytes)
