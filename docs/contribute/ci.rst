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

.. _ci_guide:

Using TVM's CI
==============

TVM uses Jenkins for running Linux continuous integration (CI) tests on
`branches <https://ci.tlcpack.ai/job/tvm/>`_ and
`pull requests <https://ci.tlcpack.ai/job/tvm/view/change-requests/>`_ through a
build configuration specified in a `Jenkinsfile <https://github.com/apache/tvm/blob/main/Jenkinsfile>`_.
Non-critical jobs run in GitHub Actions for Windows and MacOS jobs.

A standard CI run looks something like this viewed in `Jenkins' BlueOcean viewer <https://ci.tlcpack.ai/blue/organizations/jenkins/tvm/activity>`_.
CI runs usually take several hours to complete and pull requests (PRs) cannot be merged before CI
has successfully completed. To diagnose failing steps, click through to the failing
pipeline stage then to the failing step to see the output logs.

.. image:: https://github.com/tlc-pack/web-data/raw/main/images/contribute/ci.png
  :width: 800
  :alt: The Jenkins UI for a CI run


Debugging Failures
******************

When CI fails for some reason, there are several methods to diagnose the issue.

Jenkins Logs
------------

.. |pytest| replace:: ``pytest``
.. _pytest: https://docs.pytest.org/en/6.2.x/

The first place to look for a failure is in the CI logs, follow the red Xs on
the failing job to view the logs. Note:

* Jenkins does not display the full log by default, at the top of the log viewer
  is a button "Show complete log" which will take you to a plaintext version of the log
* |pytest|_ failures are summarized at the bottom of the log but you will likely
  need to scroll up to view the actual failure.

Reproduce Failures
------------------

Most TVM Python tests run under |pytest|_ and
can be run as described in :ref:`pr-testing`. For a closer environment to the one
than runs in CI you can run the docker images directly, build TVM, and execute
tests inside the container. See :ref:`docker_images` for details.

Keeping CI Green
****************

Developers rely on the TVM CI to get signal on their PRs before merging.
Occasionally breakages slip through and break ``main``, which in turn causes
the same error to show up on an PR that is based on the broken commit(s). Broken
commits can be identified `through GitHub <https://github.com/apache/tvm/commits/main>`_
via the commit status icon or via `Jenkins <https://ci.tlcpack.ai/blue/organizations/jenkins/tvm/activity?branch=main>`_.
In these situations it is possible to either revert the offending commit or
submit a forward fix to address the issue. It is up to the committer and commit
author which option to choose, keeping in mind that a broken CI affects all TVM
developers and should be fixed as soon as possible.

Skip CI for Reverts
-------------------

For reverts and trivial forward fixes, adding ``[skip ci]`` to the revert's
commit message will cause CI to shortcut and only run lint. Committers should
take care that they only merge CI-skipped PRs to fix a failure on ``main`` and
not in cases where the submitter wants to shortcut CI to merge a change faster.

.. code:: bash

  # Revert HEAD commit, make sure to insert '[skip ci]' at the beginning of
  # the commit subject
  git revert HEAD
  git checkout -b my_fix
  # After you have pushed your branch, create a PR as usual.
  git push my_repo
  # Example: Skip CI on a branch with an existing PR
  # Adding this commit to an existing branch will cause a new CI run where
  # Jenkins is skipped
  git commit --allow-empty --message "[skip ci] Trigger skipped CI"
  git push my_repo

Handling Flaky Failures
***********************

.. https://stackoverflow.com/questions/4743845/format-text-in-a-link-in-restructuredtext/4836544#4836544
.. |pytest's @xfail decorator| replace:: pytest's ``@xfail`` decorator
.. _pytest's @xfail decorator: https://docs.pytest.org/en/6.2.x/skipping.html#xfail-mark-test-functions-as-expected-to-fail
.. |strict=True| replace:: ``strict=True``
.. _strict=True: https://docs.pytest.org/en/6.2.x/skipping.html#strict-parameter

If you notice a failure on your PR that seems unrelated to your change, you should
search `recent GitHub issues related to flaky tests <https://github.com/apache/tvm/issues?q=is%3Aissue+%5BCI+Problem%5D+Flaky+>`_ and
`file a new issue <https://github.com/apache/tvm/issues/new?assignees=&labels=&template=ci-problem.md&title=%5BCI+Problem%5D+>`_
if you don't see any reports of the failure. If a certain test or class of tests affects
several PRs or commits on ``main`` with flaky failures, the test should be disabled via
|pytest's @xfail decorator|_ with |strict=True|_ and the relevant issue linked in the
disabling PR.

.. code:: python

    @pytest.mark.xfail(strict=False, reason="Flaky test: https://github.com/apache/tvm/issues/1234")
    def test_something_flaky():
        pass

``ci-docker-staging``
*********************

The `ci-docker-staging <https://github.com/apache/tvm/tree/ci-docker-staging>`_
branch is used to test updates to Docker images and ``Jenkinsfile`` changes. When
running a build for a normal PR from a forked repository, Jenkins uses the code
from the PR except for the ``Jenkinsfile`` itself, which comes from the base branch.
When branches are built, the ``Jenkinsfile`` in the branch is used, so a committer
with write access must push PRs to a branch in apache/tvm to properly test
``Jenkinsfile`` changes. If your PR makes changes to the ``Jenkinsfile``, make sure
to @ a `committer <https://github.com/apache/tvm/blob/main/CONTRIBUTORS.md>`_
and ask them to push your PR as a branch to test the changes.

.. _docker_images:

Docker Images
*************

.. |top_of_the_Jenkinsfile| replace:: top of the ``Jenkinsfile``
.. _top_of_the_Jenkinsfile: https://github.com/apache/tvm/blob/7481a297740f073b193a3f09b3e27f056e8c7f2e/Jenkinsfile#L48-L54

Each CI job runs most of its work inside a Docker container, built from files
in the `docker/ <https://github.com/apache/tvm/tree/main/docker>`_ folder. These
files are built nightly in Jenkins via the `docker-images-ci <https://ci.tlcpack.ai/job/docker-images-ci/>`_ job.
The images for these containers are hosted in the `tlcpack Docker Hub <https://hub.docker.com/u/tlcpack>`_
and referenced at the |top_of_the_Jenkinsfile|_. These can be inspected and run
locally via standard Docker commands.

.. code:: bash

    # Beware: CI images can be several GB in size
    # Get a bare docker shell in the ci-gpu container
    docker run -it tlcpack/ci-gpu:v0.78 /bin/bash

``docker/bash.sh`` will automatically grab the latest image from the ``Jenkinsfile``
and help in mounting your current directory.

.. code:: bash

    # Run the ci_cpu image specified in Jenkinsfile
    cd tvm
    bash docker/bash.sh ci_cpu
    # the tvm directory is automatically mounted
    # example: build tvm (note: this will overrwrite build/)
    $ ./tests/scripts/task_config_build_cpu.sh
    $ ./tests/scripts/task_build.sh build -j32


Reporting Issues
****************

Issues with CI should be `reported on GitHub <https://github.com/apache/tvm/issues/new?assignees=&labels=&template=ci-problem.md&title=%5BCI+Problem%5D+>`_
with a link to the relevant jobs, commits, or PRs.
