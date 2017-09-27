# CI Build Scripts

This directory contains the files and setup instructions to run all tests.

## Run locally

To run locally, we need to first install
[docker](https://docs.docker.com/engine/installation/) and
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker/wiki).

Then we can run the tasks defined in the [Jenkinsfile](../../Jenkinsfile) by
using (`ci_build.sh`)[./ci_build.sh]. For example

- lint the python codes

  ```bash
  ./ci_build.sh lint make pylint
  ```

- build codes with CUDA supports

  ```bash
  ./ci_build.sh gpu tests/scripts/task_build.sh
  ```

- do the python unittest

  ```bash
  ./ci_build.sh gpu tests/scripts/task_python_test.sh
  ```

- build the documents. The results will be available at `docs/_build/html`

  ```bash
  tests/ci_build/ci_build.sh gpu tests/scripts/task_python_docs.sh
  ```
