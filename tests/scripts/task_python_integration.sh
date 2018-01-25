#!/bin/bash
export PYTHONPATH=python:apps/extension/python
export LD_LIBRARY_PATH=lib:${LD_LIBRARY_PATH}

rm -rf python/tvm/*.pyc python/tvm/*/*.pyc

# Test TVM
make cython || exit -1

# Test extern package package
cd apps/extension
make || exit -1
cd ../..
python -m nose -v apps/extension/tests || exit -1

TVM_FFI=cython python -m nose -v tests/python/integration || exit -1
TVM_FFI=ctypes python3 -m nose -v tests/python/integration || exit -1
TVM_FFI=cython python -m nose -v tests/python/contrib || exit -1
TVM_FFI=ctypes python3 -m nose -v tests/python/contrib || exit -1

/sbin/start-stop-daemon --start --quiet --pidfile /tmp/custom_xvfb_99.pid \
        --make-pidfile --background --exec /usr/bin/Xvfb -- :99 \
        -screen 0 1400x900x24 -ac +extension GLX +render || exit -1
sleep 3
export DISPLAY=:963.0
DISPLAY=:963.0 TVM_FFI=cython python -m nose -v tests/webgl || exit -1
DISPLAY=:963.0 TVM_FFI=ctypes python3 -m nose -v tests/webgl || exit -1
