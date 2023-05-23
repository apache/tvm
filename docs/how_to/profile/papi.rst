..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.


Getting Started With PAPI
=========================

The Performance Application Programming Interface (PAPI) is a library that
provides performance counters on a variety of platforms. Performance counters
provide accurate low-level information about processors behavior during a given
execution run. This information can contain simple metrics like total cycle
count, cache misses, and instructions executed as well as more high level
information like total FLOPS and warp occupancy. PAPI makes these metrics
available while profiling.

Installing PAPI
---------------

PAPI can either be installed using your package manager (``apt-get install libpapi-dev``
on Ubuntu), or from source here:
https://bitbucket.org/icl/papi/src/master/.

Pulling the latest version of PAPI from source has caused build issues before. Therefore, it is recommended to checkout tagged version ``papi-6-0-0-1-t``.

Building TVM With PAPI
----------------------

To include PAPI in your build of TVM, set the following line in you ``config.cmake``:

.. code::

   set(USE_PAPI ON)

If PAPI is installed in a non-standard place, you can specify where it is like so:

.. code::

   set(USE_PAPI path/to/papi.pc)


Using PAPI While Profiling
--------------------------

If TVM has been built with PAPI (see above), then you can pass a
:py:class:`tvm.runtime.profiling.PAPIMetricCollector` to
:py:meth:`tvm.runtime.GraphModule.profile` to collect performance metrics. Here
is an example:

.. code:: python

    import tvm
    from tvm import relay
    from tvm.relay.testing import mlp
    from tvm.runtime import profiler_vm
    import numpy as np

    target = "llvm"
    dev = tvm.cpu()
    mod, params = mlp.get_workload(1)

    exe = relay.vm.compile(mod, target, params=params)
    vm = profiler_vm.VirtualMachineProfiler(exe, dev)

    data = tvm.nd.array(np.random.rand(1, 1, 28, 28).astype("float32"), device=dev)
    report = vm.profile(
        data,
        func_name="main",
        collectors=[tvm.runtime.profiling.PAPIMetricCollector()],
    )
    print(report)

.. code::

   Name                                    perf::CACHE-MISSES   perf::CYCLES  perf::STALLED-CYCLES-BACKEND  perf::INSTRUCTIONS  perf::STALLED-CYCLES-FRONTEND
   fused_nn_dense_nn_bias_add_nn_relu                   2,494      1,570,698                        85,608             675,564                         39,583
   fused_nn_dense_nn_bias_add_nn_relu_1                 1,149        655,101                        13,278             202,297                         21,380
   fused_nn_dense_nn_bias_add                             288        600,184                         8,321             163,446                         19,513
   fused_nn_batch_flatten                                 301        587,049                         4,636             158,636                         18,565
   fused_nn_softmax                                       154        575,143                         8,018             160,738                         18,995
   ----------
   Sum                                                  4,386      3,988,175                       119,861           1,360,681                        118,036
   Total                                               10,644      8,327,360                       179,310           2,660,569                        270,044

You can also change which metrics are collected:

.. code:: python

    report = vm.profile(
        data,
        func_name="main",
        collectors=[tvm.runtime.profiling.PAPIMetricCollector({dev: ["PAPI_FP_OPS"])],
    )

.. code::

   Name                                  PAPI_FP_OPS
   fused_nn_dense_nn_bias_add_nn_relu        200,832
   fused_nn_dense_nn_bias_add_nn_relu_1       16,448
   fused_nn_dense_nn_bias_add                  1,548
   fused_nn_softmax                              160
   fused_nn_batch_flatten                          0
   ----------
   Sum                                       218,988
   Total                                     218,988

You can find a list of available metrics by running the ``papi_avail`` and
``papi_native_avail`` commands.
