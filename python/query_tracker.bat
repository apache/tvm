set PATH=D:\workspace\cross-compiler\gcc-linaro-7.5.0-2019.12-i686-mingw32_aarch64-linux-gnu\bin\;D:\Halide\llvm-install-rel\bin\;C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64\;D:\workspace\project\nn_compiler\tvm\cmake-build-release\;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin\;%PATH%
python -m tvm.exec.query_rpc_tracker --host=192.168.6.69 --port=9190
pause