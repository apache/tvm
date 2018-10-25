Submit a Pull Request
=====================

This is a quick guide to submit a pull request, please also refer to the detailed guidelines.

- Before submit, please rebase your code on the most recent version of master, you can do it by

  .. code:: bash

    git remote add upstream [url to tvm repo]
    git fetch upstream
    git rebase upstream/master

- Make sure code style check pass by typing the following command, and all the existing test-cases pass.
  - ``docker/bash.sh tvmai/ci-lint ./tests/scripts/task_lint.sh``  
     (Note: You must install docker beforehand so you can run a docker image.)
- Add test-cases to cover the new features or bugfix the patch introduces.
- Document the code you wrote, see more at :ref:`doc_guide`
- Send the pull request,  fix the problems reported by automatic checks.
  Request code reviews from other contributors and improves your patch according to feedbacks.

  - To get your code reviewed quickly, we encourage you to help review others' code so they can do the favor in return.
  - Code review is a shepherding process that helps to improve contributor's code quality.
    We should treat it proactively, to improve the code as much as possible before the review.
    We highly value patches that can get in without extensive reviews.
  - The detailed guidelines and summarizes useful lessons.

- The patch can be merged after the reviewers approve the pull request.

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
  CACHE_PREFIX=. make -f 3rdparty/dmlc-core/scripts/packages.mk gtest

  mkdir build
  cd build
  GTEST_LIB=$TVM_ROOT/lib cmake ..
  make cpptest -j
  for test in *_test; do
    ./$test || exit -1
  done

Python
^^^^^^
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

  TVM_FFI=ctypes python -m nose -v tests/python/unittest/test_pass_storage_rewrite.py