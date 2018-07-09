# TVM Docker

This directory contains the TVM's docker infrastructure.
We use docker to quickly provide environments that can be
used to build tvm in various settings.

To run locally, we need to first install
[docker](https://docs.docker.com/engine/installation/) and
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker/).


## Use Local Build Script

Each dockerfile defines a different environment.
We use (`build.sh`)[./build.sh] to build and run the commands.
To use the docker images, we can run the following command
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
    ./docker/build.sh ci_gpu tests/scripts/task_python_unittest.sh'
      ```

- build the documents. The results will be available at `docs/_build/html`

  ```bash
    ./docker/ci_build.sh ci_gpu make -C docs html
  ```
