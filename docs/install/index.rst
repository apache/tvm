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

.. _installation:

Installing TVM
==============

.. toctree::
   :maxdepth: 2

   from_source
   docker
   nnpack

Visit the :ref:`install TVM from source <install-from-source>` page to install TVM from the source code. Installing
from source gives you the maximum flexibility to configure the build effectively from the official source releases.
If you are interested in deploying to mobile or embedded devices, you do not need to
install the entire TVM stack on your device. Instead, you only need the runtime and can install using the
:ref:`deployment and integration guide <deploy-and-integration>`.

If you would like to quickly try out TVM or run some demo and tutorials, you
can :ref:`install from Docker <docker-images>`. You can also use TVM locally through ``pip``.

.. code-block::

    # Linux/MacOS CPU build only!
    # See tlcpack.ai for other pre-built binaries including CUDA
    pip install apache-tvm

For more details on installation of pre-built binaries, visit `tlcpack.ai <https://tlcpack.ai>`_.
