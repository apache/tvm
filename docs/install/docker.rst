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
We provide several prebuilt docker images to quickly try out tvm.
These images are also helpful run through TVM demo and tutorials.
You can get the docker images via the following steps.
We need `docker <https://docs.docker.com/engine/installation/>`_ and
`nvidia-docker <https://github.com/NVIDIA/nvidia-docker/>`_ if we want to use cuda.

First, clone tvm repo to get the auxiliary scripts

.. code:: bash

    git clone --recursive https://github.com/dmlc/tvm


We can then use the following command to launch a `tvmai/demo-cpu` image.

.. code:: bash

    /path/to/tvm/docker/bash.sh tvmai/demo-cpu

You can also change `demo-cpu` to `demo-gpu` to get a CUDA enabled image.
You can find all the prebuilt images in `<https://hub.docker.com/r/tvmai/>`_


This auxiliary script does the following things:

- Mount current directory to /workspace
- Switch user to be the same user that calls the bash.sh (so you can read/write host system)
- Use the host-side network (so you can use jupyter notebook)


Then you can start a jupyter notebook by typing

.. code:: bash

   jupyter notebook


Docker Source
-------------
Check out `<https://github.com/dmlc/tvm/tree/master/docker>`_ if you are interested in
building your own docker images.
