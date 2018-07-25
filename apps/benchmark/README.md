# Performance Benchmark

## ARM CPU

### Results
Note: If a board has big.LITTLE archtiecture, we will use all big cores.
Otherwise, we will use all cores.

- **Firefly-RK3399 : 2 x Cortex A73 1.8Ghz+ 4 x Cortex A53 1.5Ghz**

```bash
$ python3 arm_cpu_imagenet_bench.py --device rk3399 --rpc-key rk3399
--------------------------------------------------
Network Name         Mean Inference Time (std dev)
--------------------------------------------------
squeezenet v1.1      44.15 ms            (0.64 ms)
mobilenet            82.23 ms            (0.67 ms)
resnet-18            168.71 ms           (0.05 ms)
vgg-16               972.03 ms           (1.75 ms)  
```

- **Raspberry Pi 3B : 4 x Cortex A53 1.2Ghz**

```bash
$ python3 arm_cpu_imagenet_bench.py --device rasp3b --rpc-key rasp3b
--------------------------------------------------
Network Name         Mean Inference Time (std dev)
--------------------------------------------------
squeezenet v1.1      94.59 ms            (0.04 ms)
mobilenet            148.82 ms           (0.18 ms)
resnet-18            347.30 ms           (0.25 ms)
vgg-16               crashed due to out of memeory
```

- **Huawei P20 Pro / Mate10 Pro (Soc: HiSilicon Kirin 970) : (4 x Cortex A73 2.36GHz + 4 x Cortex A53 1.8GHz)**

```bash
$ python3 arm_cpu_imagenet_bench.py --device p20pro --rpc-key p20pro
--------------------------------------------------
Network Name         Mean Inference Time (std dev)
-------------------------------------------------
squeezenet v1.1      29.33 ms            (0.61 ms)
mobilenet            47.47 ms            (0.65 ms)
resnet-18            84.71 ms            (0.32 ms)
vgg-16               574.62 ms           (2.14 ms)

```

- **Google Pixel 2 (Soc: Qualcomm Snapdragon 835) : (4 × Kyro 2.35 GHz, 4 × Kyro 1.9 GHz)**

```bash
$ python3 arm_cpu_imagenet_bench.py --device pixel2 --rpc-key pixel2
--------------------------------------------------
Network Name         Mean Inference Time (std dev)
--------------------------------------------------
squeezenet v1.1      27.74 ms            (0.41 ms)
mobilenet            42.05 ms            (0.08 ms)
resnet-18            67.28 ms            (0.05 ms)
vgg-16               427.75 ms           (8.58 ms)
```

### How to run

1. Start an RPC Tracker on the host machine
```bash
python3 -m tvm.exec.rpc_tracker
```

2. Register your device to the tracker
* For Linux device
  * Build tvm runtime on your device [Help](https://docs.tvm.ai/tutorials/cross_compilation_and_rpc.html#build-tvm-runtime-on-device).
  * Register your device to tracker by
  ```bash
  python3 -m tvm.exec.rpc_sever --tracker=[HOST_IP]:9190 --key=[DEVICE_KEY]
  ```
  replace `[HOST_IP]` with the IP address of the host machine, `[DEVICE_KEY]` with the name of device.
  
  E.g. For my RK3399, I use `python3 -m tvm.exec.rpc_sever --tracker=10.77.1.123:9190 --key=rk3399`

* For Andoird device
   * Build and install tvm rpc apk on your device [Help](https://github.com/dmlc/tvm/tree/master/apps/android_rpc).
     Make sure you can pass the android rpc test. Then you have alreadly known how to register.

3. Verify the device registration  
  We can query all registered devices by
  ```bash
  python3 -m tvm.exec.query_rpc_tracker
  ```
  You should be able to find your devices in `Queue Status`. Make sure
  the registration is correct before go ahead.

  For our test environment, one sample output can be 
  ```bash
  Queue Status                
  ------------------------------
  key            free    pending    
  ------------------------------
  mate10pro      1       0   
  p20pro         2       0  
  pixel2         2       0 
  rk3399         2       0
  rasp3b         8       0
  ```
 4. Run benchmark  
  We did auto-tuning for the above devices, and release pre-tuned
  parameters in [this repo](https://github.com/uwsaml/tvm-distro).
  During compilation, TVM will download these operator parameters automatically.

  But we don't tune for other devices, so you can only run benchmark for these devices.
  ```bash
  python3 arm_cpu_imagenet_bench.py --device rasp3b --rpc-key rasp3b
  python3 arm_cpu_imagenet_bench.py --device rk3399 --rpc-key rk3399
  python3 arm_cpu_imagenet_bench.py --device pixel2 --rpc-key pixel2
  python3 arm_cpu_imagenet_bench.py --device p20pro --rpc-key p20pro
  python3 arm_cpu_imagenet_bench.py --device mate10pro --rpc-key mate10pro  
  ```
  
  If you do not do tuning and run the benchmark for other devices directly,
  the performance is not gauranteed (This is still doable, you can pick a most
  similar device and reuse its parameter).
  In order to get the best performance, you need to tune for you own device,
  please follow [tutorial](404.html).

