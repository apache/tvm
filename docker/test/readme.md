The dockerfile in this directory is apply to deploy and test TVM environments (including CPU and NVIDIA_GPU) easily.

### 1. TVM on CPU
Please install Docker. The docker of TVM_CPU occupies 3.5G hard disk space.
(1) Dockerfile of TVM_CPU is shown in "Dockerfile_cpu"

(2) Enter the docker env
```
$ docker build -f Dockerfile.CPU -t test_tvm_cpu . --network host
$ docker run -it --rm --name my_tvm_cpu test_tvm_cpu:latest tail -f /dev/null 
$ docker exec -it test_tvm_cpu bash
```

(3) Run the inference code "tune_network_x86_test.py". Please mount the inference code in the path "/workspace/".

The output will be shown as followed:
```
root@4b9ef6056c35:/workspace/# python3 tune_network_x86_test.py
Get model...
...100%, 0.02 MB, 0 KB/s, 271 seconds passed
conv2d NHWC layout is not optimized for x86 with autotvm.
One or more operators have not been tuned. Please tune your model for better performance. Use DEBUG logging level to see more details.
conv2d NHWC layout is not optimized for x86 with autotvm.
Evaluate inference time cost...
Mean inference time (std dev): 200.63 ms (0.07 ms)
```

### 2. TVM on NVIDIA_GPU

Please install Docker and Nvidia-Docker firstly. The docker of TVM_CPU occupies 6.2G hard disk space.
(1) Dockerfile of NVIDIA_GPU is shown in "Dockerfile_nvidia_gpu".

(2) Enter the docker env
```
$ nvidia-docker build -f Dockerfile.NVIDIA_GPU -t test_tvm_nvidia_gpu . --network host
$ nvidia-docker run -it --rm --name test_tvm_nvidia_gpu  test_tvm_nvidia_gpu:latest tail -f /dev/null 
$ nvidia-docker exec -it test_tvm_nvidia_gpu  bash
```

(3) Run the inference code "inference_cuda_test.py". Please mount the inference code in the path "/workspace/".

The output will be shown as followed:

```
root@0b9abfe951a9:/workspace/# python3 tune_relay_cuda_test.py
Extract tasks...
...100%, 0.47 MB, 592 KB/s, 0 seconds passed
One or more operators have not been tuned. Please tune your model for better performance. Use DEBUG logging level to see more details.
Evaluate inference time cost...
Mean inference time (std dev): 11.20 ms (0.06 ms)
root@0b9abfe951a9:/workspace/#
```

### 3. Pull from docker-hub
For convenience, we can pull the pre-built docker images from docker hub directly.
```
$ docker pull huangyongtao/test_tvm_cpu
$ docker pull huangyongtao/test_tvm_nvidia_gpu
```
Then, we can mount the python inference code into the docker for test or deployment directly.

