Perform Code Reviews
====================

This is a general guideline for code reviewers. First of all, while it is great to add new features to a project, we must also be aware that each line of code we introduce also brings **technical debt** that we may have to eventually pay.

Open source code is maintained by a community with diverse backend, and it is even more important to bring clear, documented and maintainable code. Code reviews are shepherding process to spot potential problems, improve quality of the code. We should, however, not rely on code review process to get the code into a ready state. Contributors are encouraged to polish the code to a ready state before requesting reviews. This is especially expected for code owner and comitter candidates.

Here are some checklists for code reviews, it is also helpful reference for contributors


Hold the Highest Standard
-------------------------
The first rule for code reviewers is to always keep the highest standard, and do not approve code just to "be friendly". Good, informative critics each other learn and prevents technical debt in early stages.

Ensure Test Coverage
--------------------
Each new change of features should introduce test cases, bug fixes should include regression tests that prevent the problem from happening again.

Documentations are Mandatory
----------------------------
Documentation is usually a place we overlooked, new functions or change to a function should be directly updated in documents. A new feature is meaningless without documentation to make it accessible. See more at :ref:`doc_guide`

Deliberate on User-facing API
-----------------------------
A good, minimum and stable API is critical to the project’s life. A good API makes a huge difference. Always think very carefully about all the aspects including naming, arguments definitions and behavior. One good rule to check is to be consistent with existing well-known package’s APIs if the feature overlap. For example, tensor operation APIs should always be consistent with the numpy.

Minimum Dependency
------------------
Always be cautious in introducing dependencies. While it is important to reuse code and not reinventing the wheel, dependencies can increase burden of users in deployment. A good design principle only depends on the part when a user actually use it.

Ensure Readability
------------------
While it is hard to implement a new feature, it is even harder to make others understand and maintain the code you wrote. It is common for a PMC or committer to not being able to understand certain contributions. In such case, a reviewer should say "I don’t understand" and ask the contributor to clarify. We highly encourage code comments which explain the code logic along with the code.

Concise Implementation
----------------------
Some basic principles applied here: favor vectorized array code over loops, is there existing API that solves the problem.

Document Lessons in Code Reviews
--------------------------------
When you find there are some common lessons that can be summarized in the guideline,
add it to the :ref:`code_guide`.
It is always good to refer to the guideline document when requesting changes,
so the lessons can be shared to all the community.

Respect each other
------------------
The code reviewers and contributors are paying the most precious currencies in the world -- time. We are volunteers in the community to spend the time to build good code, help each other, learn and have fun hacking.

Learn from other Code Reviews
-----------------------------
There can be multiple reviewers reviewing the same changes. Many cases other reviewers
may spot things you did not find. Try to learn from other code reviews,
when possible, document these lessons.

Approve and Request Changes Explicitly
--------------------------------------
The contributor and code owner can request code reviews from multiple reviewers.
Remember to approve changes when your comments are addressed in a code review.
To do so -- please click on changes tab in the pull request, then select approve,
or comment on the code and click request changes.
Code owner can decide if the code can be merged in case by case if some of the reviewers
did not respond in time(e.g. a week) and existing reviews are sufficient.
