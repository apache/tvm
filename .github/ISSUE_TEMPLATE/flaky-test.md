---
name: "\U00002744 Flaky Test"
about: Report a flaky test, make sure to include links to CI runs, a sample failure log, and the name of the test(s)
title: "[Flaky Test] "
labels: "test: flaky"
---

Thanks for participating in the TVM community! We use https://discuss.tvm.ai for any general usage questions and discussions. The issue tracker is used for actionable items such as feature proposals discussion, roadmaps, and bug tracking. You are always welcomed to post on the forum first :smile_cat:

These tests were found to be flaky (intermittently failing on `main` or failed in a PR with unrelated changes). As per [the docs](https://github.com/apache/tvm/blob/main/docs/contribute/ci.rst#handling-flaky-failures), these failures will be disabled in a PR that references this issue until the test owners can fix the source of the flakiness.

### Test(s)

- `tests/python/some_file.py::the_test_name`

### Jenkins Links

- Please provide link(s) to failed CI runs. If runs are for a PR, explain why your PR did not break the test (e.g. did not touch that part of the codebase)
