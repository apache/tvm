from . import _ffi_api

def generate(ir):
    return _ffi_api.GenerateCPP(ir)
