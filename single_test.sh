make -j8

rm -rf python/tvm/*.pyc python/tvm/*/*.pyc python/tvm/*/*/*.pyc

TVM_FFI=ctypes python -m pytest -v tests/python/relay/test_external_codegen.py