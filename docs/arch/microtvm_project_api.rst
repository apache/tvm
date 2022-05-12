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

.. _microtvm_project_api:

microTVM Project API
====================

About microTVM Project API
--------------------------

The microTVM Project API allows TVM to automatically run models on
unconventional or embedded platforms. It allows platforms to define a standard
function to integrate TVM compiler output with boilerplate platform-specific
code, producing a runnable **Project**. Project API then further defines
functions to build that project, program compatible devices accessible from the
TVM machine, and communicate with the running code so that TVM can perform
host-driven inference and autotuning.

There are many cases where it might be desirable simply to invoke microTVM as a
tool from your platform's build process. Indeed, for the average firmware
developer, this is likely to be all they need. However, there are a couple of
use cases when you may want to teach microTVM how to build firmware using your
platform's build tool:

1.  To enable AutoTVM and AutoScheduling on your platform. Defining a Project
    API implementation allows TVM to tune models for peak performance on your
    platform.
2.  To enable engineers without firmware expertise to experiment with models on
    your platform. Defining a Project API implementation allows these engineers
    to leverage the standard TVM Python workflows to perform host-driven
    inference on your platform.
3.  Integration Testing. Defining a Project API implementation allows you to
    create Continuous Integration Tests which verify model correctness and
    performance on your platform.

API Definition
--------------

The full API is the ``abstractmethod`` defined on ``ProjectAPIHandler`` in
`python/tvm/micro/project_api/server.py <https://github.com/apache/tvm/blob/main/python/tvm/micro/project_api/server.py>`_.
Rather than duplicate the documentation here, we simply refer you to that class.

How TVM uses Project API
------------------------

This section explains how the Project API should be used with TVM. Project API
is defined around the *Project* as the buildable unit of firmware. TVM expects
to be provided initially with a directory containing a *Template Project*, which
together with a :ref:`Model Library Format <model_library_format>` file can be
built into a runnable project.

Inside the Template Directory is (typically) a Python script implementing the
API server. TVM launches this script in a subprocess and sends commands to the
server to perform each of the actions outlined above.

The typical usage flow is as follows:

1. Launch Project API server in Template Project.
2. Verify the API server is version-compatible with TVM, plus read properties
   of the implementation, by sending ``server_info_query`` command.
3. Generate a new project by sending command ``generate_project`` to create a
   new project. The arguments to this command is a Model Library Format and a
   non-existent directory which should be populated with the generated
   project. The Template Project API server should copy itself into the
   newly-generated project.
4. Terminate the Template Project API server.
5. Launch Project API server in Generated Project.
6. Verify the API server is version-compatible with TVM, plus read properties
   of the implementation, by sending ``server_info_query`` command.
7. Build and flash the projec by sending commands ``build`` and ``flash`` to the
   API server.
8. Communicate with the target. Send command ``open_transport`` followed by
   commands ``write_transport`` and ``read_transport`` to write and read from
   e.g. a serial port attached to the target. Upon completion,
   ``close_transport`` is sent.
9. Terminate Project API server.

Disk Layout of the Project
--------------------------

In the root directory of a project (template or generated), one of the following
two files must exist:

- ``microtvm_api_server.py`` - the suggested approach. Place a
  python3-compatible Python script in the root directory. TVM will execute this
  script in its own process using the same interpreter used to execute TVM.
- ``microtvm_api_server.sh`` (on Windows, ``microtvm_api_server.bat``) -
  alternate approach. When a different Python interpreter is necessary, or
  when you want to implement the server in a different language, create this
  executable file. TVM will launch this file in a separate process.

Aside from these two files, no other restrictions are made on the layout.

Communication between TVM and Project API Server
------------------------------------------------

TVM communicates with the Project API server using `JSON-RPC 2.0
<https://www.jsonrpc.org/specification>`_. TVM always launches API servers using
the following command-line:

``microtvm_api_server.py --read-fd <n> --write-fd <n>``

Commands are sent from TVM to the server over the file descriptor given by
``--read-fd`` and replies are received by TVM from the server over the file
descriptor given by ``--write-fd``.

Helpers for Implementing the API server in Python
-------------------------------------------------

TVM provides helper utilities that make it easy to implement the server in Python.
To implement the server in Python, create ``microtvm_api_server.py`` and add
``from tvm.micro.project_api import server`` (or, copy this file into your template
project--there are no dependencies--and import it there instead). Next, subclass
``ProjectAPIHander``::

    class Handler(server.ProjectAPIHandler):
        def server_info_query(self, tvm_version):
            # Implement server_info_query

        def generate_project(self, model_library_format_path, standalone_crt_dir, project_dir, options):
            # Implement generate_project

        # ...

Finally, invoke the helper ``main()``::

    if __name__ == "__main__":
        server.main(Handler())

Using Project API from ``tvmc``
-------------------------------

Each major Project API command is available through the ``tvmc micro``
sub-command to make debugging interactions simple. Invoke ``tvmc micro --help``
for more information.
