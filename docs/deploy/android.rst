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

Deploy to Android
=================

Build model for Android Target
------------------------------

Relay compilation of model for android target could follow same approach like android_rpc.
The code below will save the compilation output which is required on android target.


.. code:: python

    lib.export_library("deploy_lib.so", ndk.create_shared)
    with open("deploy_graph.json", "w") as fo:
        fo.write(graph.json())
    with open("deploy_param.params", "wb") as fo:
        fo.write(relay.save_param_dict(params))

deploy_lib.so, deploy_graph.json, deploy_param.params will go to android target.

TVM Runtime for Android Target
------------------------------

Refer `here <https://github.com/apache/incubator-tvm/blob/master/apps/android_deploy/README.md#build-and-installation>`_ to build CPU/OpenCL version flavor TVM runtime for android target.
From android java TVM API to load model & execute can be referred at this `java <https://github.com/apache/incubator-tvm/blob/master/apps/android_deploy/app/src/main/java/org/apache/tvm/android/demo/MainActivity.java>`_ sample source.
