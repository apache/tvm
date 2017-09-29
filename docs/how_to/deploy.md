How to Deploy TVM Modules
=========================
We provide an example on how to deploy TVM modules in [apps/howto_deploy](https://github.com/dmlc/tvm/tree/master/apps/howto_deploy)

To run the example, you can use the following command

```bash
cd apps/howto_deploy
./run_example.sh
```

Get TVM Runtime Library
-----------------------

![](http://www.tvmlang.org/images/release/tvm_flexible.png)

The only thing we need is to link to a TVM runtime in your target platform.
TVM provides a minimum runtime, which costs around 300K to 600K depending on how much modules we use.
In most cases, we can use ```libtvm_runtime.so``` that comes with the build.

If somehow you find it is hard to build ```libtvm_runtime```, checkout [tvm_runtime_pack.cc](https://github.com/dmlc/tvm/tree/master/apps/howto_deploy/tvm_runtime_pack.cc).
It is an example all in one file that gives you TVM runtime.
You can compile this file using your build system and include this into your project.

You can also checkout [apps](https://github.com/dmlc/tvm/tree/master/apps/) for example applications build with TVM on iOS, Android and others.

Dynamic Library vs. System Module
---------------------------------
TVM provides two ways to use the compiled library.
You can checkout [prepare_test_libs.py](https://github.com/dmlc/tvm/tree/master/apps/howto_deploy/prepare_test_libs.py)
on how to generate the library and [cpp_deploy.cc](https://github.com/dmlc/tvm/tree/master/apps/howto_deploy/cpp_deploy.cc) on how to use them.

- Store library as a shared library and dynamically load the library into your project.
- Bundle the compiled library into your project in system module mode.

Dynamic loading is more flexible and can load new modules on the fly. System module is a more ```static``` approach.  We can use system module in places where dynamic library loading is banned.
