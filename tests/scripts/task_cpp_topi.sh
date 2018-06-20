export PYTHONPATH=python:topi/python

python -m nose -v topi/tests/python_cpp || exit -1
python3 -m nose -v topi/tests/python_cpp || exit -1
