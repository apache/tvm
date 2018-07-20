# Deploy to Android


## Build model for Android Target

NNVM compilation functions of model for android target could follow same approach like android_rpc.

An reference scripts can be found at [save_android_model_functions.py](https://github.com/dayanandasiet/TVM_models/blob/master/save_android_model_functions.py) which help to save deploy_lib.so, deploy_graph.json, deploy_param.params for target android phone.

## TVM Runtime for Android Target

Refer [here](https://github.com/dmlc/tvm/blob/master/apps/android_deploy/README.md#build-and-installation) to build CPU/OpenCL/Vulkan version flavor TVM runtime for android target.
From android java TVM API to load model & execute can be refered at this [java](https://github.com/dmlc/tvm/blob/master/apps/android_deploy/app/src/main/java/ml/dmlc/tvm/android/demo/MainActivity.java) sample source.
