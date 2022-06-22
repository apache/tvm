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

.. contents::
  :local:

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

Most TVM Python tests run under |pytest|_ and can be run as described in :ref:`pr-testing`.


Reporting Issues
****************

Issues with CI should be `reported on GitHub <https://github.com/apache/tvm/issues/new?assignees=&labels=&template=ci-problem.md&title=%5BCI+Problem%5D+>`_
with a link to the relevant jobs, commits, or PRs.
