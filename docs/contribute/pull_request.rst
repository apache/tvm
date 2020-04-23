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

Submit a Pull Request
=====================

This is a quick guide to submit a pull request, please also refer to the detailed guidelines.

- Before submit, please rebase your code on the most recent version of master, you can do it by

  .. code:: bash

    git remote add upstream [url to tvm repo]
    git fetch upstream
    git rebase upstream/master

- Make sure code style check pass by typing the following command, and all the existing test-cases pass.
- ``docker/bash.sh tvmai/ci-lint ./tests/scripts/task_lint.sh``. (Note: You must install docker beforehand so you can run a docker image.)
- Add test-cases to cover the new features or bugfix the patch introduces.
- Document the code you wrote, see more at :ref:`doc_guide`
- Send the pull request and fix the problems reported by automatic checks.
- Request code reviews from other contributors and improves your patch according to feedbacks.

  - To get your code reviewed quickly, we encourage you to help review others' code so they can do the favor in return.
  - Code review is a shepherding process that helps to improve contributor's code quality.
    We should treat it proactively, to improve the code as much as possible before the review.
    We highly value patches that can get in without extensive reviews.
  - The detailed guidelines and summarizes useful lessons.

- The patch can be merged after the reviewers approve the pull request.


CI Environment
--------------
We use docker container to create stable CI environments
that can be deployed to multiple machines.
You can find the prebuilt images in `<https://hub.docker.com/r/tvmai/>`_ .
Because we want a relatively stable CI environment and make use of pre-cached image,
all of the CI images are built and maintained by committers.

Upgrade of CI images can cause problems and need fixes to accommodate the new env.
Here is the protocol to update CI image:

- Send PR to upgrade build script in the repo
  - Can be done by a contributor, the following steps need committership.
- Build the new docker image
- Tag the docker image with a new version and push to tvmai
- Update the version(most of the time increase the minor version) in the Jenkinsfile, send a PR.
- Fix any issues wrt to the new image versions in the PR.
- Merge the PR and now we are in new version.
- Tag the new version as the latest.
- Periodically cleanup the old versions on local workers

Testing
-------
Even though we have hooks to run unit tests automatically for each pull request, It's always recommended to run unit tests
locally beforehand to reduce reviewers' burden and speedup review process.

C++
^^^
.. code:: bash

  # assume you are in tvm source root
  TVM_ROOT=`pwd`

  # you need to install google test first, gtest will be installed to $TVM_ROOT/lib
  apt-get install -y libgtest-dev
  CACHE_PREFIX=. make -f 3rdparty/dmlc-core/scripts/packages.mk gtest

  mkdir build
  cd build
  GTEST_LIB=$TVM_ROOT/lib cmake -DUSE_LLVM=ON ..
  make cpptest -j$(nproc)
  for test in *_test; do
    ./$test
  done

Python
^^^^^^
Necessary dependencies:

.. code:: bash

  pip install --user pytest Cython

If you want to run all tests:

.. code:: bash

  # build tvm
  make

  ./tests/scripts/task_python_unittest.sh

If you want to run a single test:

.. code:: bash

  # build tvm
  make

  # let python know where to find tvm related libraries
  export PYTHONPATH=python:topi/python
  rm -rf python/tvm/*.pyc python/tvm/*/*.pyc python/tvm/*/*/*.pyc

  TVM_FFI=ctypes python -m pytest -v tests/python/unittest/test_pass_storage_rewrite.py

  # Additionally if you want to run a single test, for example test_all_elemwise inside a file.
  TVM_FFI=ctypes python -m pytest -v -k "test_all_elemwise" tests/python/frontend/tflite/test_forward.py
