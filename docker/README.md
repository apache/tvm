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
- Mount current directory to /workspace and set it as home
- Switch user to be the same user that calls the bash.sh
- Use the host-side network

The helper bash script can be useful to build demo sessions.

## Prebuilt Docker Images

We provide several pre-built images for doing quick exploration with TVM installed.
For example, you can run the following command to get ```tvmai/demo-cpu``` image.

```bash
/path/to/tvm/docker/bash.sh tvmai/demo-cpu
```

Then inside the docker container, you can type the following command to start the jupyter notebook
```bash
jupyter notebook
```

Check out https://hub.docker.com/r/tvmai/ to get the full list of available prebuilt images.


## Use Local Build Script

We also provide script to build docker images locally.
We use (`build.sh`)[./build.sh] to build and run the commands.
To build and run docker images, we can run the following command
at the root of the project.

```bash
./docker/build.sh image_name [command]
```

Here image_name corresponds to the docker defined in the
```Dockerfile.image_name```.

You can also start an interactive session by typing

```bash
./docker/build.sh image_name -it bash
```

The build command will map the tvm root to /workspace/ inside the container
with the same user as the user invoking the docker command.
Here are some common use examples to perform CI tasks.

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
  ./docker/build.sh ci_cpu make -C golang tests
  ```
