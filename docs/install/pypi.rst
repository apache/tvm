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

.. _install-from-pypi:

Install from PyPI
=================

For most Python users, the quickest way to get started is to install the Apache
TVM wheel from PyPI:

.. code-block:: bash

   pip install apache-tvm

This installs the Python package, including modules such as ``tvm.tirx``, and
is suitable for trying tutorials that do not require a custom build.

For more details on installing the TIRx compiler and optional kernel library,
visit the :doc:`TIRx installation </tirx/install>` page. If you need to
customize TVM's build configuration, visit the
:ref:`install TVM from source <install-from-source>` page instead.
