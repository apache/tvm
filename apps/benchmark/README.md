# Performance Benchark

## ARM CPU

### How to run

### Results
If a board is big.lITTLE archtiecture, we will use big cores only.

* Firefly-RK3399 : 2 x Cortex A73 1.8Ghz+ 4 x Cortex A53 1.5Ghz

```bash
$ python3 arm_cpu_imagenet_bench.py --device rk3399 --rpc-key rk3399
--------------------------------------------------
Network Name         Mean Inference Time (std dev)
--------------------------------------------------
squeezenet v1.1      44.15 ms            (0.64 ms)
mobilenet            82.23 ms            (0.67 ms)
resnet-18            168.71 ms           (0.05 ms)
vgg-16               969.63 ms           (0.75 ms)  
```

* Raspberry Pi 3B : 4 x Cortex A53 1.2Ghz

```bash
$ python3 arm_cpu_imagenet_bench.py --device rasp3b --rpc-key rasp3b
--------------------------------------------------
Network Name         Mean Inference Time (std dev)
--------------------------------------------------
squeezenet v1.1      93.59 ms            (0.04 ms)
mobilenet            147.82 ms           (0.18 ms)
resnet-18            347.30 ms           (0.25 ms)
```

* Huawei P20 Pro / Mate10 Pro (Soc: HiSilicon Kirin 970) : (4 x Cortex A73 2.36GHz + 4 x Cortex A53 1.8GHz) 

```bash
$ python3 arm_cpu_imagenet_bench.py --device p20pro --rpc-key p20pro
```

* Google Pixel 2 (Soc: Qualcomm Snapdragon 835) : (4 × Kyro 2.35 GHz, 4 × Kyro 1.9 GHz)

```bash
$ python3 arm_cpu_imagenet_bench.py --device pixel2 --rpc-key pixel2
```

