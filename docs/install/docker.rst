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

.. _docker-images:

Docker Images
=============
We provide docker utility scripts to help developers to setup development environment.
They are also helpful run through TVM demo and tutorials.
We need `docker <https://docs.docker.com/engine/installation/>`_ and
`nvidia-docker <https://github.com/NVIDIA/nvidia-docker/>`_ if we want to use cuda.

Get a tvm source distribution or clone the github repo to get the auxiliary scripts

.. code:: bash

    git clone --recursive https://github.com/apache/incubator-tvm tvm


We can then use the following command to launch a docker image.

.. code:: bash

    /path/to/tvm/docker/bash.sh <image-name>

Here the image-name can be a local docker image name, e.g. `tvm.ci_cpu` after you have done
the local build. Or a pre-built third party image (`tvmai/demo-cpu` or `tvmai/ci-gpu`).

This auxiliary script does the following things:

- Mount current directory to /workspace
- Switch user to be the same user that calls the bash.sh (so you can read/write host system)
- Use the host-side network on Linux. Use the bridge network and expose port 8888 on macOS,
  because host networking driver isn't supported. (so you can use jupyter notebook)


Then you can start a jupyter notebook by typing

.. code:: bash

   jupyter notebook

You might see an error ``OSError: [Errno 99] Cannot assign requested address`` when starting
a jupyter notebook on macOS. You can change the binding IP address by

.. code:: bash

   jupyter notebook --ip=0.0.0.0

Note that on macOS, because we use bridge network, jupyter notebook will be reportedly running
at an URL like ``http://{container_hostname}:8888/?token=...``. You should replace the ``container_hostname``
with ``localhost`` when pasting it into browser.

You can find some un-official prebuilt images in `<https://hub.docker.com/r/tvmai/>`_.
Note that these are convenience images and are not part of the ASF release.

Docker Source
-------------
Check out `The docker source <https://github.com/apache/incubator-tvm/tree/master/docker>`_ if you are interested in
building your own docker images.
