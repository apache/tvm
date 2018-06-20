Submit a Pull Request
=====================

This is a quick guide to submit a pull request, please also refer to the detailed guidelines.

- Before submit, please rebase your code on the most recent version of master, you can do it by

  .. code:: bash

    git remote add upstream [url to tvm repo]
    git fetch upstream
    git rebase upstream/master

- Make sure code style check pass by typing ``make lint``, and all the existing test-cases pass.
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
