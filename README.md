# CPrune: Compiler-Informed Model Pruning for Efficient Target-Aware DNN Execution

Our source code is based on an open deep learning compiler stack Apache TVM (https://github.com/apache/tvm) and Microsoft nni (https://github.com/microsoft/nni).

# Abstract
Mobile devices run deep learning models for various purposes, such as image classification and speech recognition. Due to the resource constraints of mobile devices, researchers have focused on either making a lightweight deep neural network (DNN) model using model pruning or generating an efficient code using compiler optimization. Surprisingly, we found that the straightforward integration between model compression and compiler auto-tuning often does not produce the most efficient model for a target device. We propose CPrune, a compiler-informed model pruning for efficient target-aware DNN execution to support an application with a required target accuracy. CPrune makes a lightweight DNN model through informed pruning based on the structural information of subgraphs built during the compiler tuning process. Our experimental results show that CPrune increases the DNN execution speed up to 2.73x compared to the state-of-the-art TVM auto-tune while satisfying the accuracy requirement.

# Motivation
![example_v2](https://user-images.githubusercontent.com/47049810/177085650-eab0d6fd-bf73-47ae-8931-ad5f325abc29.png)
An experiment to find the fastest model whose accuracy is higher than 92.80% accuracy for the target device among various pruned models of VGG-16 for CIFAR-10. The best model with pruning does not guarantee the best after compiler optimization. For example, the best model with pruning achieves only 2174 FPS (figures per second) while the suboptimal one after pruning obtains a higher 2857 FPS after compiler optimization.

# Overview
![overview_rev11](https://user-images.githubusercontent.com/47049810/177086104-6727c3a7-208c-4004-af6a-02dce96cdc9f.png)
The goal of CPrune is to find and prune neurons that largely impact the execution of the DNN model on the target device while meeting the accuracy requirements. This informed pruning effectively prevents the issue of the best pruning model not being the best on the target device after compilation.

The above depicts how CPrune leverages the information collected during compiler optimization to prune neurons from the DNN model effectively. The shaded boxes indicate what CPrune adds to the existing compiler optimization framework. A DNN model consists of many convolutional layers, each of which is represented as a subgraph (1). A subgraph is assigned to a task, and multiple same subgraphs can point to the same task. Then, a DNN compiler creates numerous intermediate representations for each task and selects the fastest program on the target device (2). After compiling the aggregate of the fastest programs comprising a DNN model, CPrune checks if the model meets the minimum execution and accuracy requirements. 
After that, CPrune sorts tasks based on their execution times in descending order to find the most efficient model with further pruning. Since a task of a longer execution time could reduce the execution time of a model significantly, CPrune selects the most time-consuming task as a candidate for further pruning (3).
CPrune now needs to know which subgraph(s) is associated with this task as pruning candidates. To preserve the program structure of the fastest execution, we also store the fastest program for each task. For this purpose, CPrune builds a table keeping the relationship among tasks, subgraphs, and programs (4). Finally, CPrune prunes subgraphs of the selected task while ensuring their code structures follow the structure of the fastest program of that task (5). Since the computation structure impacts the execution time, preserving the same computation structure after pruning is critical for efficient pruning. This process continues to make the most efficient pruned DNN model satisfying the accuracy requirement.

# Experimental Results
<img width="650" alt="table1" src="https://user-images.githubusercontent.com/47049810/177086838-9aff036c-550f-455a-be1d-bdc723160d9b.png">
We compare CPrune with other pruning schemes on different mobile platforms, as shown the above table. CPrune shows a higher FPS than the model-based pruning models (e.g., PQF, FPGM, AMC). CPrune also shows similar or better performance than the hardware-aware pruning model (e.g., NetAdapt) with TVM. It also shows that indirect metrics such as FLOPS and parameters do not fully reflect actual performance. While FLOPS is a suitable indirect measure of the extent of compression obtained during the pruning process, FPS suitably reflects the pruning gains in terms of execution times or speeds.

# How to set up
## Host PC side
### Install nvidia-container-runtime
1. Add package repository \
      curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | sudo apt-key add - \
      distribution=$(. /etc/os-release; echo $ID$VERSION_ID) \
      curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | \ <br/>
      sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list \
      sudo apt-get update
2. Package installation \
      sudo apt-get install -y nvidia-container-runtime
3. Installation check \
      which nvidia-container-runtime-hook
### Build and run
4. Check /docker/install/ubuntu_install_python.sh to fit with 'python3.6' <br>
5. docker build -t tvm3.demo_android -f docker/Dockerfile.demo_android ./docker
6. docker run --pid=host -h tvm3 -v $PWD:/workspace -w /workspace -p 9192:9192 --name tvm3 -it --gpus all tvm3.demo_android bash

## Docker side
7. Check if the GPU driver works properly \
      nvidia-smi
### Anaconda install
8. Download the Anaconda installation script \
      wget -P /tmp https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh <br>
9. Run the script to start the installation process \
      bash /tmp/Anaconda3-2020.02-Linux-x86_64.sh \
      source ~/.bashrc
### Conda environment
10. conda create -n nni ptyhon=3.6 <br>
11. conda activate nni <br>
12. conda install -c anaconda cudnn <br>
13. conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
### Build the TVM
14. mkdir build <br>
    cd build <br>
    cmake -DUSE_LLVM=llvm-config-8 -DUSE_RPC=ON -DUSE_VULKAN=ON -DUSE_GRAPH_EXECUTOR=ON .. <br>
    make -j10
### Install Android TVM RPC - Install Gradle
15. sudo apt install curl zip vim <br>
    curl -s "https://get.sdkman.io" | bash <br>
    source "$HOME/.sdkman/bin/sdkman-init.sh" <br>
    sdk install gradle 6.8.3
### Install TVM4J - Java Frontend for TVM Runtime
16. cd /workspace <br>
    make jvmpkg <br>
    pip3 install decorator <br>
    (Optional) sh tests/scripts/task_java_unittest.sh <br>
    make jvminstall
### ~/.bashrc
17. echo 'export PYTHONPATH=/workspace/python:/workspace/vta/python:${PYTHONPATH}' >> ~/.bashrc <br>
    echo 'export ANDROID_HOME=/opt/android-sdk-linux' >> ~/.bashrc <br>
    echo 'export TVM_NDK_CC=/opt/android-toolchain-arm64/bin/aarch64-linux-android-g++' >> ~/.bashrc <br>
    echo 'export TF_CPP_MIN_LOG_LEVEL=1' >> ~.bashrc <br>
    sudo apt-get install libjemalloc1 <br>
    echo 'export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.1' >> ~/.bashrc <br>
    source ~/.bashrc <br>
    conda activate nni
### Create a standalone toolchain
18. cd /opt/android-sdk-linux/ndk/21.3.6528147/build/tools/ <br>
    ./make-standalone-toolchain.sh --platform=android-28 --use-llvm --arch=arm64 --install-dir=/opt/android-toolchain-arm64
### tvmrpc-release.apk for CPU
19-1. cd /workspace/apps/android_rpc/app/src/main/jni/ <br>
      vim ./config.mk # ADD_C_INCLUDES = /opt/adrenosdk-linux-5_0/Development/Inc (depending on the phone)
### tvmrpc-release.apk for GPU
19-2. Get libOpenCL.so file from your phone to Host PC <br>
	    adb pull /system/vendor/lib64/libOpenCL.so ./ <br>
      Put the libOpenCL.so to /workspace/apps/android_rpc/app/src/main/jni/ <br>
      mv config.mk cpu_config.mk <br>
      mv gpu_config.mk config.mk
### Build APK (to create an apk file)
20. cd /workspace/apps/android_rpc <br>
    gradle clean build <br>
    ./dev_tools/gen_keystore.sh # generate a signature <br>
    ./dev_tools/sign_apk.sh # get the signed apk file <br>
    Upload app/build/outputs/apk/release/tvmrpc-release.apk file to the Android device and install it
### Additional stuff
21. pip3 install nni colorama tornado json-tricks schema scipy PrettyTable psutil xgboost cloudpickle absl-py tensorboard tensorflow pytest
### Basic docker setup
22. exit <br>
    docker start tvm3 <br>
    docker exec -it tvm3 bash <br>
    conda activate nni
### RPC tracker    
23. python3 -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9192
### RPC connection check
24. (new terminal) python3 -m tvm.exec.query_rpc_tracker --host=0.0.0.0 --port=9192


# How to execute
In tutorials/frontend, there are two core files main.py and c_pruner.py. <br>
You can select an input model and type the accuracy requirement in main.py, and run the file.
If you want to look at the CPrune algorithm code, please look at c_pruner.py.
