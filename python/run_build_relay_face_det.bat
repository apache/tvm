call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
set PYTHONPATH=D:\workspace\project\nn_compiler\tvm\python;%PYTHONPATH%
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\bin\;D:\workspace\cross-compiler\gcc-linaro-7.5.0-2019.12-i686-mingw32_aarch64-linux-gnu\bin\;D:\Halide\llvm-install-rel\bin\;D:\workspace\project\nn_compiler\tvm\cmake-build-release\;%PATH%
python D:\workspace\project\nn_compiler\tvm\python\build_from_relay.py --path D:\workspace\project\nn_compiler\tvm\python\Deploy --model_name face_det --input_name input --input_size 1 3 480 640 --device arm_cuda --export_path D:\workspace\project\nn_compiler\tvm\python\Deploy --fp16 True --eval True
pause