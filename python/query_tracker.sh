export PATH=/home/share/data/workspace/project/nn_compiler/tvm/cmake-build-release_clang:$PATH
python -m tvm.exec.query_rpc_tracker --host=192.168.6.252 --port=9190
pause