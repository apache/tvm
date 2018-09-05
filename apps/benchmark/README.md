# Performance Benchmark

## Results

See results on wiki page https://github.com/dmlc/tvm/wiki/Benchmark

## How to Reproduce

To obtain the best performance, we always do auto-tuning for the specific devices and get
the parameters for used kernels. We release some pre-tuned networks on some devices
so users can easily reproduce our results. TVM will download related tuning cache files
during compilation.

If you don't have the following listed devices, you can still run these scripts.
You can pick the one that is most similar to your device as argument.
In general, the performance should also be good.

For your custom devices and networks, we recommend that you tune it by yourself.
Please follow the tutorial for
[NVIDIA GPU](https://docs.tvm.ai/tutorials/autotvm/tune_nnvm_cuda.html),
[ARM CPU](https://docs.tvm.ai/tutorials/autotvm/tune_nnvm_arm.html),
[Mobile GPU](https://docs.tvm.ai/tutorials/autotvm/tune_nnvm_mobile_gpu.html).

### NVIDIA GPU

Build TVM with LLVM and CUDA enabled. [Help](https://docs.tvm.ai/install/from_source.html)

```bash
python3 gpu_imagenet_benchmark.py --model 1080ti
python3 gpu_imagenet_benchmark.py --model titanx
```

### AMD GPU

Build TVM with LLVM and ROCm enabled. [Help](https://docs.tvm.ai/install/from_source.html)
```bash
python3 gpu_imagenet_benchmark.py --model gfx900 --target rocm
```

### ARM CPU & Mali GPU
For embedded deivces, we use RPC infrastructure in TVM to make the management easy.
So you need to use it for reproducing benchmark results.

0. Build TVM with LLVM enabled. [Help](https://docs.tvm.ai/install/from_source.html)

1. Start an RPC Tracker on the host machine
```bash
python3 -m tvm.exec.rpc_tracker
```

2. Register devices to the tracker
* For Linux device
  * Build tvm runtime on your device [Help](https://docs.tvm.ai/tutorials/nnvm/deploy_model_on_rasp.html#build-tvm-runtime-on-device)
  * Register your device to tracker by
  ```bash
  python3 -m tvm.exec.rpc_sever --tracker=[HOST_IP]:9190 --key=[DEVICE_KEY]
  ```
  replace `[HOST_IP]` with the IP address of the host machine, `[DEVICE_KEY]` with the name of device.
  
  E.g. Here is an example command for RK3399,
  `python3 -m tvm.exec.rpc_sever --tracker=10.77.1.123:9190 --key=rk3399`, where 10.77.1.123 is the IP address of the tracker.

* For Android device
   * Build and install tvm RPC apk on your device [Help](https://github.com/dmlc/tvm/tree/master/apps/android_rpc).
     Make sure you can pass the android rpc test. Then you have alreadly known how to register.

3. Verify the device registration  
  We can query all registered devices by
  ```bash
  python3 -m tvm.exec.query_rpc_tracker
  ```
  You should be able to find your devices in `Queue Status`. Make sure the registration is correct before going ahead.

  For our test environment, one sample output can be 
  ```bash
  Queue Status                
  ----------------------------------
  key          total  free  pending    
  ----------------------------------
  mate10pro    1      1     0
  p20pro       2      2     0 
  pixel2       2      2     0
  rk3399       2      2     0
  rasp3b       8      8     0
  ```

4. Run benchmark  
  ```bash
  # ARM CPU
  python3 arm_cpu_imagenet_bench.py --model rasp3b --rpc-key rasp3b
  python3 arm_cpu_imagenet_bench.py --model rk3399 --rpc-key rk3399
  python3 arm_cpu_imagenet_bench.py --model pixel2 --rpc-key pixel2
  python3 arm_cpu_imagenet_bench.py --model p20pro --rpc-key p20pro
  python3 arm_cpu_imagenet_bench.py --model mate10pro --rpc-key mate10pro  

  # Mali GPU
  python3 mobile_gpu_imagenet_bench.py --model rk3399 --rpc-key rk3399
  ```
