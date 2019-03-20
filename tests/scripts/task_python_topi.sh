export PYTHONPATH=python:topi/python

# Rebuild cython
make cython || exit -1
make cython3 || exit -1

rm -rf python/tvm/*.pyc python/tvm/*/*.pyc python/tvm/*/*/*.pyc
rm -rf topi/python/topi/*.pyc topi/python/topi/*/*.pyc topi/python/topi/*/*/*.pyc topi/python/topi/*/*/*/*.pyc 

python -m nose -v topi/tests/python || exit -1
python3 -m nose -v topi/tests/python || exit -1
