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

.. _committer_guide:

Committer Guide
===============
This is an evolving document to provide some helpful tips for committers.
Most of them are lessons learned during development.
We welcome every committer to contribute to this document.
See the :ref:`community_guide` for an overview of
the committership and the general development process.

Community First
---------------
The collective effort of the community moves the project forward and
makes the project awesome for everyone.
When we make a decision, it is always helpful to keep the community in mind.
Here are some example questions that we can ask:

- How can I encourage new contributors to get more involved in the project?
- Can I help to save my fellow committers' time?
- Have I enabled the rest of the community to participate the
  design proposals?


Public Archive Principle
------------------------
While private channels such as face to face discussion are useful for development,
they also create barriers for the broader community's participation.
The Apache way of development requires all decisions
to be made in public channels, which are archived and accessible to everyone.
As a result, any contributor can keep up with the development by watching the
archives and join the development anytime.

While this principle applies to every contributor,
it is especially important for committers.
Here are some example applications of this principle:

- When getting a project-related question from a personal channel,
  encourage the person to open a public thread in the discuss forum,
  so others in the community can benefit from the answer.
- After an in-person discussion, send a summary to public channels
  (as an RFC or a discuss thread).


Shepherd a Pull Request
-----------------------

Here are some tips to shepherd a pull request.
You can also take a look at the :ref:`code_review_guide`.

- Assign the PR to yourself, so that other committers
  know that the PR has already been tended to.
- Make use of the status label to indicate the current status.
- Check if an RFC needs to be sent.
- If the contributor has not requested a reviewer, kindly
  ask the contributor to do so.
  If the PR comes from a new contributor,
  help the contributor to request reviewers
  and ask the contributor to do so next time.
- Moderate the reviews, ask reviewers to approve explicitly.
- Mark the PR as accepted and acknowledge the contributor/reviewers.
- Merge the PR :)


Time Management
---------------
There are many things that a committer can do, such as
moderating discussions, pull request reviews and
code contributions.

Working on an open source project can be rewarding,
but also be a bit overwhelming sometimes.
A little bit of time management might be helpful to alleviate the problem.
For example, some committers have a "community day" in a week
when they actively manage outstanding PRs,
but watch the community less frequently in the rest of the time.

Remember that your merit will never go away, so please
take your time and pace when contributing to the project:)


Broad Collaboration
-------------------
Sometimes, we tend to only interact with people we know.
However, broad collaborations are necessary to the success of the project.
Try to keep that in mind, shepherd PRs for, and request code reviews from
community members who you do not interact physically.


Keeping CI Green
----------------
Developers rely on the TVM CI to get signal on their PRs before merging.
Occasionally breakges slip through and break ``main``, which in turn causes
the same error to show up on an PR that is based on the broken commit(s).
In these situations it is possible to either revert the offending commit or
submit a forward fix to address the issue. It is up to the committer and commit
author which option to choose, keeping in mind that a broken CI affects all TVM
developers and should be fixed as soon as possible.

For reverts and trivial forward fixes, adding ``[skip ci]`` to the revert's
commit message will cause CI to shortcut and only run lint. Committers should
take care that they only merge CI-skipped PRs to fix a failure on ``main`` and
not in cases where the submitter wants to shortcut CI to merge a change faster.

.. code:: bash

  # Example: Skip CI on a revert
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

