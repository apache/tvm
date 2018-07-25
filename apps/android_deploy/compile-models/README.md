# TVM Compiled Models for Android Phone using Darknet frontend

### Processor Architecture
Find the target android phone processor architecture using ABI (Application Binary Interface) properties though [ADB](https://developer.android.com/studio/command-line/adb) shell.
```sh
$ adb shell getprop ro.product.cpu.abi
arm64-v8a
```
| Name on Droid Hardware Info/ABI | Processor |
| ------ | ------ |
| ARMv7 or armeabi | arm |
| AArch64 or arm64 | arm64 |
| x86 or x86abi | x86 |

### Set Build environment varibale
Set build environment variable which includes processor architecture (arch), tvm build optimization level (opt_level), compile flavor version (exec_flavor), compile required model (model_name).
```bash
arch = "arm64"
opt_level = 0
exec_flavor = "vulkan"
model_name = 'extraction'
```

### Run python script to save compiled model function

```sh
$ python save_android_model_functions.py
Downloading from url https://github.com/siju-samuel/darknet/blob/master/cfg/extraction.cfg?raw=true to extraction.cfg
...100%, 0 MB, 2 KB/s, 2 seconds passed
Downloading from url http://pjreddie.com/media/files/extraction.weights?raw=true to extraction.weights
...100%, 89 MB, 750 KB/s, 122 seconds passed
Downloading from url https://github.com/siju-samuel/darknet/blob/master/lib/libdarknet.so?raw=true to libdarknet.so
...100%, 0 MB, 161 KB/s, 3 seconds passed
Loading weights from ./extraction.weights...Done!
Converting darknet to nnvm symbols...
Compiling the model...
You can still compile vulkan module but cannot run locally
Saving the compiled nnvm model functions...
```