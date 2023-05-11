..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

..  Boilerplate script for installing Zephyr in the microTVM
    tutorials that use it. Does not show up as a separate file
    on the documentation website.


Install microTVM Python dependencies
------------------------------------

TVM does not include a package for Python serial communication, so
we must install one before using microTVM. We will also need TFLite
to load models.

    .. code-block:: bash

        %%shell
        pip install pyserial==3.5 tflite==2.1
