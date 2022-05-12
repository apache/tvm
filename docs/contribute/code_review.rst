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

.. _code_review_guide:


Code Reviews
============

.. contents::
  :depth: 2
  :local:

Open source code is maintained by a community with diverse backgrounds, interests, and goals.
Hence it is important to provide clear, documented and maintainable code and processes. Code reviews are a
shepherding process used to collectively spot potential problems, improve quality of the code, and educate both contributors
and reviewers about the code base and its assumptions. It is also one mechanism to ensure there are multiple people who can
maintain a related piece of code together. Contributors are encouraged to polish the code to a reviewable
state before requesting reviews. This is especially important for committer candidates, as committers are expected
to participate in not only writing code but also reviewing it.

This document is a living guideline for code review in open source. Please also take sometime to read
:ref:`community_guide` about the general development process.

Building Trust
--------------

First and foremost, we are building a community that based on trust, which takes time
and effort to both build and maintain.  We expect our community members to work together in a
constructive way and work together with common sense. Although we all have different sets of backgrounds,
interests and goals we must work together to find solutions that work for the larger community.
Trust-based collaboration is also a key tenant of the Apache way and an important factor to consider in growing the community,
and promoting members to official roles.

Community Participation
-----------------------

Everyone is welcomed to comment on PRs. We encourage committers to wait for some period of time(e.g. three days)
before merging PR that contains a major architecture change. The goal is to give people time to speak up and
express interest in reviewing and participate.

Remembering that we are all coming from different backgrounds is important here. For example some community members
work in different time zones, only work on open source during work hours, or may be traveling or having other events
going on in their lives. An important part of working in a large project is ensuring there is collective understanding,
so no one person is a bottleneck. While it is important to allow time for participation in code review we also can not
block all changes on all reviewers. Remember that helping people land PRs is a great way to encourage broader
participation, especially for those who volunteer their time to contribute.

Part of this is trusting and communicating with fellow maintainers that if changes need to be applied in the future
that PR authors will later follow through on their promises. It is the responsibility of committers to listen to all
feedback whether from PMC members or new contributors and consider what actions need to be taken.

Read the code carefully
-----------------------

Sometimes we may quickly read through the code and only pick up on a selective aspects of the code. These type of comments
are usually helpful and should be welcomed in the community. However,  they are only part of performing code review and
should be part of more comprehensive feedback. A good and careful code review is a large time investment and sometimes
can be longer than writing the actual contribution.

For example receiving only highly critical feedback on minor aspects of your PR rarely feels good, and it can be discouraging
if your time and effort was not reciprocated during review. Practicing empathy when acting both as a contributor and committer
is important and can help make you a more effective code reviewer and contributor.

We expect that all committers carefully read and understand the code before signing off. There is a lot of trust involved when
a committer hits the merge button. In the meantime, we acknowledge that sometimes problems slip through, in that case, the
merger is responsible for ensuring the correct follow up actions are taken.

Be Respectful
-------------

- To everyone who are making comments: making constructive comment will help new contributors to land their PRs
  timely and help us welcome new members to the community.

- To authors: reviewers should spend significant time reading the code, and a careful review could be as time intensive
  as writing the code from scratch. Respectfully address review comments and reciprocate the review by helping review
  others changes in the future.

Most importantly focus on having a constructive conversation, and try to assume best intentions when interacting as a reviewer.
If there is something in the process not working, consider getting some face time with the other contributors and discussing
how to improve the process or communication.

Factors to Consider about Code Quality
--------------------------------------

High quality code is critical to the long term success of the project. There are many factors of code quality to consider
during a code review:

- F0: Overall architecture. This includes the definition of public modules, key data structures and public interfaces.
  Good architectural choices are critical to the success of the project in the long run.
- F1: Architectural consistency. There are usually multiple ways to implement a new feature. We must ensure new
  features are consistent with previous overall architectural choices and interact well with the existing code.
  Every new feature increases the complexity of the project, and a consistent design ideally minimizes the increase
  in complexity bought by a new feature, making it easier to maintain code in the long run.
- F2: Code robustness and test coverage. Ensure code runs correctly in all possible settings(platforms), ensure
  test coverage of the new feature. Clear error messages for user facing errors.
- F3: User facing API documentation: documentation of public user facing APIs and key module interfaces are mandatory.
  This includes the API, data structures that appears in the public interface (i.e., `include/tvm` and user facing python APIs).
  We generally encourage well documented code and include some form of documentations for internal APIs that are used in
  multiple places, see also F4.
- F4: Code readability. Readability involves multiple aspects: instructive and consistent function names, clear implementation
  of the overall flow, descriptive comments for complex code logic and internal functions. Readable code is easier to maintain.

Architectural design and consistency are the most important factors since they are likely to introduce the most long term technical debt.
As a result, committers should most carefully consider these factors before merging the code.

Test coverage and API documentation are expected for code contributions.

Code readability is relatively a subjective matter compared to the other ones.
Different people have different thoughts on how to best write code. Reviewers should make constructive and actionable comments.
In the meantime, code review should not be used as a way to get others to write code exactly the way you would.
Conversely you should also consider that what you may easily understand, or find acceptable might not work for the larger
community or other members. Use your judgment on what is appropriate based on the content and the scope of the contribution
and where the contributor is coming from.

We follow common :ref:`code_guide` when writing code. Style guides help ensure that code is readable and maintainable by others,
long after the original author has moved on. Style guides are more than about code formatting — they also pertain
to the correct way to document code, variable naming, and other conventions that are not enforced by automatic formatters.

Consensus Building
------------------

Disagreements can happen during code reviews. We encourage building consensus among the people involved. We are working together
and building trust with each other in OSS. The nature of OSS means sometimes we make compromises on less significant issues to
make steady progress and welcome broader participation in the community. Compromise unfortunately means sometimes the world will
not be exactly as we would like, this true even for leaders of the community.

- Be civil and build consensus through constructive technical-based conversations.
- A committer who owns the area can serve as a shepherd to drive the discussion by taking all the conversations into consideration,
  and suggest a resolution with to move forward.
- Because a lot of trust is involved on the committer(shepherd), they should read the PR carefully before sign off. Additionally,
  the merger should also take the responsibility to followup in case there are problems caused by the merge.

Consistency
-----------

A final remark is that we are all human and its hard to always be perfectly consistent. If contributors feel that you didn't apply these guidelines
in a consistent way it is important to listen and hear folks out. We will constantly have to iterate on processes and guidelines as we evolve as a community.
Our goal is to strive to be consistent and objective but all of us are unfortunately human and imperfect and will need to adjust and learn.

Additional Recommendations
--------------------------

Deliberate on API and Data Structures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A minimum and stable API is critical to the project’s life. A good API makes a huge difference. Always think very carefully about all the aspects including naming, argument definitions and behavior.

When possible, pay more attention still to the proposed API design during code reviews.
Remember, it is easier to improve code implementation, but it is extremely hard to change an API once accepted.
We should treat data structures that are shared across modules(e.g. AST) in the same way.
If/when uncertain, start a conversation with more developers before committing.

Here are some useful principles for designing APIs:

- Be consistent with existing well-known package’s APIs if the features overlap.
  For example, tensor operation APIs should always be consistent with the numpy API.
- Be consistent with existing APIs in the same project.
  For example, we should use the same argument ordering across all the optimization passes,
  so there is no "surprise" when using them.
- Think about whether the API will change in the future.
  For example, we will have more options like loop_unrolling and device placement policy
  as we add more optimizations in build. We can package optimization knobs into a build
  configuration object. In this way, the build API is stable over time, even though it may be enriched.
- Write documentation. Documentation is mandatory for APIs and sometimes writing documents helps
  us to think further about the design as well as whether we need to add further clarifications.
- Minimum. Think about how many lines of code a user has to write to use the API.
  Remove layers of abstraction when possible.

Minimize Dependencies
~~~~~~~~~~~~~~~~~~~~~
Always be cautious in introducing dependencies. While it is important to reuse code and avoid reinventing the wheel,
dependencies can increase burden of users in deployment. A good design principle is that a feature or function
should only have a dependency if/when a user actually use it.

Concise Implementation
~~~~~~~~~~~~~~~~~~~~~~
Some basic principles applied here: favor vectorized array code over loops, use existing APIs that solve the problem.

Document Lessons in Code Reviews
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When you find there are some common or recurring lessons that can be summarized,
add it to the :ref:`code_guide`.
It is always good to refer to the guideline document when requesting changes,
so the lessons can be shared to all the community.


Learn from other Code Reviews
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
There can be multiple reviewers reviewing the same changes. Many times other reviewers
may spot things you did not find. Try to learn from other code reviews, when possible, document these lessons.

Approve and Request Changes Explicitly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The contributor and code owner can request code reviews from multiple reviewers.
Remember to approve changes when your comments are addressed in a code review.
To do so -- please click on changes tab in the pull request, then select approve,
or comment on the code and click request changes.
Code owner can decide if the code can be merged in case by case if some of the reviewers
did not respond in time(e.g. a week) and existing reviews are sufficient.

Reviewers
~~~~~~~~~
Reviewers should strive to leave timely feedback on pull requests for which their
review was requested. Reviewing code is an important part of the project's health
and should be considered a regular responsibility for contributors. Automated
tooling helps out in this regard, as PRs with no activity for a set amount of
time will get a bot comment pinging the relevant parties.
