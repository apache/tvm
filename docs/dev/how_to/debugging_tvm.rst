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

.. _debugging-tvm:

Debugging TVM
==============

**NOTE**: This page is a work in-progress. Everyone is welcomed to add suggestions and tips via
sending a PR to modify this page. The goal with this page is to centralize the commonly-used
techniques being used to debug TVM and to spread awareness to the community. To that end, we may
seek to promote more broadly-used techniques to the top of this doc.

VLOGging
--------

TVM provides a verbose-logging facility that allows you to commit trace-level debugging messages
without impacting the binary size or runtime of TVM in production. You can use VLOG in your code
as follows:

.. code-block:: c++

    void Foo(const std::string& bar) {
      VLOG(2) << "Running Foo(" << bar << ")";
      // ...
    }

In this example, the integer ``2`` passed to ``VLOG()`` indicates a verbosity level. The higher the
level, the more logs printed. In general, TVM levels range from 0 to 2, with 3 being used only for
extremely low-level core runtime properties. The VLOG system is configured at startup time to print
VLOG statements between ``0`` and some integer ``N``. ``N`` can be set per-file or globally.

VLOGs don't print or impact binary size or runtime by default (when compiled with proper
optimization). To enable VLOGging, do the following:

1. In ``config/cmake``, ensure you ``set(USE_RELAY_DEBUG ON)``. This flag is used to enable
   VLOGging.
2. Launch Python passing ``TVM_LOG_DEBUG=<spec>``, where ``<spec>>`` is a comma-separated list of
   level assignments of the form ``<file_name>=<level>``. Here are some specializations:

    - The special filename ``DEFAULT`` sets the VLOG level setting for all files.
    - ``<level>>`` can be set to ``-1`` to disable VLOG in that file.
    - ``<file_name>`` is the name of the c++ source file (e.g. ``.cc``, not ``.h``) relative to the
      ``src/`` directory in the TVM repo. You do not need to supply ``src/`` when specifying the
      file path, but if you do, VLOG will still interpret the path correctly.

Examples:

.. code-block:: shell

   # enable VLOG(0), VLOG(1), VLOG(2) in all files.
   $ TVM_LOG_DEBUG=DEFAULT=2 python3 -c 'import tvm'

   # enable VLOG(0), VLOG(1), VLOG(2) in all files, except not VLOG(2) in src/bar/baz.cc.
   $ TVM_LOG_DEBUG=DEFAULT=2,bar/baz.cc=1 python3 -c 'import tvm'

   # enable VLOG(0), VLOG(1), VLOG(2) in all files, except not in src/foo/bar.cc.
   $ TVM_LOG_DEBUG=DEFAULT=2,src/foo/bar.cc=-1 python3 -c 'import tvm'
