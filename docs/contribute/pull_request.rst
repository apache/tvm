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

.. contents::
  :depth: 2
  :local:

Guidelines
----------

- We recommend authors send well scoped PRs that are easy to review and revert in case there is a problem. As such, authors should avoid merging multiple unrelated changes into a single PR
- Before you submit a PR, please rebase your code on the most recent version of ``main``, you can do it by
  running

  .. code:: bash

    git remote add upstream [url to tvm repo]
    git fetch upstream
    git rebase upstream/main

- Make sure code passes lint checks

    .. code:: bash

      # While the lint commands used should be identical to those run in CI, this command reproduces
      # the CI lint procedure exactly (typically helpful for debugging lint script errors or
      # to avoid installing tools manually)
      python tests/scripts/ci.py lint

      # Run all lint steps.
      docker/lint.sh

      # To run steps individually, specify their step names on the command-line. An incorrectly
      # spelled step name causes the tool to print all available steps.
      docker/lint.sh <step_name> ...

    If the clang-format lint check fails, run git-clang-format as follows to automatically reformat
    your code:

    .. code:: bash

      # Run clang-format check for all the files that changed since upstream/main
      docker/bash.sh ci_lint ./tests/lint/git-clang-format.sh upstream/main

- Add test-cases to cover the new features or bugfix the patch introduces.
- Document the code you wrote, see more at :ref:`doc_guide`
- `Create a pull request <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request>`_ and fix the problems reported by CI checks.
- Request code reviews from other contributors and improve your patch according to their reviews by ``@``-ing them in your pull request. Tags in PR titles will automatically tag subscribed users, so make sure to put relevant topics in your PR titles (e.g. ``[microTVM] a cool change`` and not ``a cool change for microTVM``).

  - To get your code reviewed quickly, we encourage you to help review others' code so they can do the favor in return.
  - Code review is a shepherding process that helps to improve contributor's code quality.
    We should treat it proactively, to improve the code as much as possible before the review.
    We highly value patches that can get in without extensive reviews.
  - The detailed guidelines and summarizes useful lessons.

- The PR can be merged after the reviewers approve the pull request.

CI Environment
--------------
We use Docker images to create stable CI environments that can be deployed to multiple machines.
Follow the steps in `this issue template <https://github.com/apache/tvm/issues/new?assignees=&labels=&template=ci-image.md&title=%5BCI+Image%5D+>`_
to update a CI Docker image.

.. _pr-testing:

Testing
-------
Even though we have hooks to run unit tests automatically for each pull request, it's always recommended to run unit tests
locally beforehand to reduce reviewers' burden and speedup review process.

Docker (recommended)
^^^^^^^^^^^^^^^^^^^^
``tests/scripts/ci.py`` replicates the CI environment locally and provides a user-friendly interface.
The same Docker images and scripts used in CI are used directly to run tests. It also deposits builds
in different folders so you can maintain multiple test environments without rebuilding from scratch
each time (e.g. you can test a change in CPU and i386 while retaining incremental rebuilds).

.. code:: bash

    # see all available platforms
    python tests/scripts/ci.py --help
    python tests/scripts/ci.py cpu --help

    # run the CPU build in the ci_cpu docker container (build will be left in
    # the build-cpu/ folder)
    # note: the CPU and GPU Docker images are quite large and may take some
    # time to download on their first use
    python tests/scripts/ci.py cpu

    # run the CPU build in the ci_cpu docker container and then run unittests
    python tests/scripts/ci.py cpu --unittest

    # quickly iterate by running a specific test and skipping the rebuild each time
    python tests/scripts/ci.py cpu --skip-build --tests tests/python/unittest/test_tir_transform_inject_rolling_buffer.py::test_upscale

    # run the CPU build and drop into a shell in the container
    python tests/scripts/ci.py cpu --interactive

We regularly update our docker images and, over time, stale images may unnecessarily consume disk
space. You can remove stale images that aren't used in the presently checked-out branch plus any
other worktrees using the following command:

.. code:: bash

    docker/clear-stale-images.sh

Consult the ``--help`` for more options.

C++ (local)
^^^^^^^^^^^

Running the C++ tests requires installation of gtest, following the instructions in
:ref:`install-from-source-cpp-tests`


.. code:: bash

  # assume you are in tvm source root
  TVM_ROOT=`pwd`

  ./tests/scripts/task_cpp_unittest.sh

Python (local)
^^^^^^^^^^^^^^
Necessary dependencies:

.. code:: bash

  pip install --user pytest Cython synr

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
  export PYTHONPATH=python
  rm -rf python/tvm/*.pyc python/tvm/*/*.pyc python/tvm/*/*/*.pyc

  TVM_FFI=ctypes python -m pytest -v tests/python/unittest/test_pass_storage_rewrite.py

  # Additionally if you want to run a single test, for example test_all_elemwise inside a file.
  TVM_FFI=ctypes python -m pytest -v -k "test_all_elemwise" tests/python/frontend/tflite/test_forward.py
