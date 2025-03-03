<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# TVM Docker

This directory contains the TVM's docker infrastructure.
We use docker to provide build environments for CI and images for demo.
We need [docker](https://docs.docker.com/engine/installation/) and
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker/) for GPU images.

## Start Docker Bash Session

You can use the following helper script to start an
interactive bash session with a given image_name.

```bash
/path/to/tvm/docker/bash.sh image_name
```

The script does the following things:

- Mount current directory to the same location in the docker container, and set it as home
- Switch user to be the same user that calls the bash.sh
- Use the host-side network

The helper bash script can be useful to build demo sessions.

## Prebuilt Docker Images

You can use third party pre-built images for doing quick exploration with TVM installed.
For example, you can run the following command to launch ```tvmai/demo-cpu``` image.

```bash
/path/to/tvm/docker/bash.sh tvmai/demo-cpu
```

Then inside the docker container, you can type the following command to start the jupyter notebook
```bash
jupyter notebook
```

You can find some un-official prebuilt images in https://hub.docker.com/r/tlcpack/ .
Note that these are convenience images and are not part of the ASF release.


## Use Local Build Script

We also provide script to build docker images locally.
We use (`build.sh`)[./build.sh] to build and (optionally) run commands
in the container. To build and run docker images, we can run the following
command at the root of the project.

```bash
./docker/build.sh image_name [command(optional)]
```

Here image_name corresponds to the docker defined in the
```Dockerfile.image_name```.

You can also start an interactive session by typing

```bash
./docker/build.sh image_name -it bash
```

The built docker images are prefixed by ``tvm.``, for example the command

````bash
./docker/build.sh image_name
````

produces the image ``tvm.ci_cpu`` that is displayed in the list of docker images
using the command ``docker images``. To run an interactive terminal, execute:

````bash
./docker/bash.sh tvm.ci_cpu
````

or

````bash
./docker/bash.sh tvm.ci_cpu echo hello tvm world
````

the same applies to the other images (``./docker/Dockerfile.*```).

The command ``./docker/build.sh image_name COMMANDS`` is almost equivelant to
``./docker/bash.sh image_name COMMANDS`` but in the case of ``bash.sh``
a build attempt is not done.

The build command will map the tvm root to the corresponding location
inside the container with the same user as the user invoking the
docker command.  Here are some common use examples to perform CI
tasks.

- lint the python codes

  ```bash
  ./docker/build.sh ci_lint make pylint
  ```

- build codes with CUDA support

  ```bash
  ./docker/build.sh ci_gpu make -j$(nproc)
  ```

- do the python unittest

  ```bash
  ./docker/build.sh ci_gpu tests/scripts/task_python_unittest.sh
  ```

- build the documents. The results will be available at `docs/_build/html`

  ```bash
  ./docker/ci_build.sh ci_gpu make -C docs html
  ```

- build golang test suite.

  ```bash
  ./docker/build.sh ci_cpu tests/scripts/task_golang.sh
  ```
