#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Tests for gen_requirements, found in python/."""

import collections
import contextlib
import os
import sys

import tvm
import tvm.testing

import pytest

# Insert the parent dir to python/tvm into the import path, so that gen_requirements may be
# imported.
sys.path.insert(0, os.path.dirname(tvm.__file__))
try:
    import gen_requirements
finally:
    sys.path.pop(0)


@contextlib.contextmanager
def patch(obj, **kw):
    old = {}
    for prop_name, new in kw.items():
        old[prop_name] = getattr(obj, prop_name)
        setattr(obj, prop_name, new)
    yield
    for prop_name, value in old.items():
        setattr(obj, prop_name, value)


PROBLEM_REQUIREMENTS = [
    ("extras-pre-core", ("", ["foo", 123])),  # entry before core
    (456, ("", ["foo", "bar"])),  # invalid extras name, deps should not be processed
    ("core", ("", ["foo"])),  # ordinary core entry.
    ("wrong-description-type", (None, ["foo"])),  # wrong description type
    ("bad-value", None),  # value field is not a 2-tuple
    ("bad-value-2", ("", ["foo"], 34)),  # value field is not a 2-tuple
    ("invalid", ("", ["qux"])),  # duplicate invalid entry, all items valid.
    ("extras-foo", ("", ["bar", "baz"])),  # ordinary extras entry.
    ("invalid", ("", ["baz", None, 123])),  # valid extra name, invalid deps.
    ("unsorted", ("", ["qux", "bar", "foo"])),  # deps out of order
    ("versioned_dep", ("", ["baz==1.2", "foo==^2.0", "buz<3", "bar>4"])),
    ("duplicate_dep", ("", ["buz", "buz", "foo"])),  # duplicate listed dependency
    ("dev", ("", ["baz", "qux"])),  # ordinary dev entry.
    ("extras-post-dev", ("", ["bar", "buzz"])),  # entry after dev
]


def test_validate_requirements():
    with patch(gen_requirements, REQUIREMENTS_BY_PIECE=None):
        assert gen_requirements.validate_requirements_by_piece() == [
            "must be list or tuple, see None"
        ]

    with patch(gen_requirements, REQUIREMENTS_BY_PIECE=PROBLEM_REQUIREMENTS):
        problems = gen_requirements.validate_requirements_by_piece()
        assert problems == [
            'piece extras-pre-core: must list after "core" (core must be first)',
            "piece extras-pre-core: deps should be a list of strings, got ['foo', 123]",
            "piece 456: must be str",
            "piece wrong-description-type: description should be a string, got None",
            (
                'piece bad-value: should be formatted like ("bad-value", ("<requirements.txt '
                'comment>", ["dep1", "dep2", ...])). got: None'
            ),
            (
                'piece bad-value-2: should be formatted like ("bad-value-2", '
                '("<requirements.txt comment>", ["dep1", "dep2", ...])). got: (\'\', '
                "['foo'], 34)"
            ),
            "piece invalid: listed twice",
            "piece invalid: deps should be a list of strings, got ['baz', None, 123]",
            "piece unsorted: deps must be sorted. Correct order:\n  ['bar', 'foo', 'qux']",
            "piece versioned_dep: deps must be sorted. Correct order:\n  ['bar>4', 'baz==1.2', 'buz<3', 'foo==^2.0']",
            "piece versioned_dep: dependency baz==1.2 should not specify a version. Add it to CONSTRAINTS instead.",
            "piece versioned_dep: dependency foo==^2.0 should not specify a version. Add it to CONSTRAINTS instead.",
            "piece versioned_dep: dependency buz<3 should not specify a version. Add it to CONSTRAINTS instead.",
            "piece versioned_dep: dependency bar>4 should not specify a version. Add it to CONSTRAINTS instead.",
            "piece duplicate_dep: dependency buz listed twice",
            'piece extras-post-dev: must list before "dev" (dev must be last)',
            'pieces other than "core" and "dev" must appear in alphabetical order: '
            "['bad-value', 'bad-value-2', 'duplicate_dep', 'extras-foo', 'extras-post-dev', "
            "'extras-pre-core', 'invalid', 'invalid', 'unsorted', 'versioned_dep', "
            "'wrong-description-type']",
        ]


TEST_REQUIREMENTS_BY_PIECE = (
    ("core", ("core tvm requirements", ("bar", "foo", "non-constrained"))),
    ("extra-one", ("requirements for one feature", ("baz", "qux"))),
    ("extra-two", ("requirements for two feature", ("buz", "qux", "semver-minor", "semver-patch"))),
    ("dev", ("requirements for dev", ("buz", "oof", "rab"))),
)


def test_validate_constraints():
    with patch(
        gen_requirements,
        REQUIREMENTS_BY_PIECE=TEST_REQUIREMENTS_BY_PIECE,
        CONSTRAINTS=(
            ("unlisted", "~=3"),
            ("double-specified", "<2"),
            (
                "double-specified",
                "==3",
            ),
            ("bad-constraint", "1.2.0"),
            ("bad-semver-constraint", "i don't match the regex :P"),
            ("alpha-semver-constraint", "^foo.bar.23"),
        ),
    ):
        problems = gen_requirements.validate_constraints()
        assert problems == [
            "unlisted: not specified in REQUIREMENTS_BY_PIECE",
            "double-specified: not specified in REQUIREMENTS_BY_PIECE",
            "double-specified: specified twice",
            "double-specified: not specified in REQUIREMENTS_BY_PIECE",
            "bad-constraint: not specified in REQUIREMENTS_BY_PIECE",
            'bad-constraint: constraint "1.2.0" does not look like a valid constraint',
            "bad-semver-constraint: not specified in REQUIREMENTS_BY_PIECE",
            'bad-semver-constraint: constraint "i don\'t match the regex :P" does not look like a valid constraint',
            "alpha-semver-constraint: not specified in REQUIREMENTS_BY_PIECE",
            "alpha-semver-constraint: invalid semver constraint ^foo.bar.23",
            "CONSTRAINTS entries should be in this sorted order: ['alpha-semver-constraint', 'bad-constraint', 'bad-semver-constraint', 'double-specified', 'double-specified', 'unlisted']",
        ]


TEST_CONSTRAINTS = (
    ("bar", "==1.0"),
    ("baz", ">2.3"),
    ("buz", "^1.3.0"),
    ("non-constrained", None),  # Support a comment.
    ("oof", "==0.3.4"),
    ("qux", "~=1.2.4"),
    ("semver-minor", "^0.2.2-patch2.post3+buildmeta"),  # Ensure prerelease and buildmeta preserved.
    ("semver-patch", "^0.0.2+bm"),  # Ensure postrelease preserved.
)


def test_join_requirements():
    with patch(
        gen_requirements,
        REQUIREMENTS_BY_PIECE=TEST_REQUIREMENTS_BY_PIECE,
        CONSTRAINTS=TEST_CONSTRAINTS,
    ):
        requirements = gen_requirements.join_requirements()
        assert requirements == collections.OrderedDict(
            [
                ("core", ("core tvm requirements", ["bar==1.0", "foo", "non-constrained"])),
                ("extra-one", ("requirements for one feature", ["baz>2.3", "qux~=1.2.4"])),
                (
                    "extra-two",
                    (
                        "requirements for two feature",
                        [
                            "buz>=1.3.0,<2.0.0",
                            "qux~=1.2.4",
                            "semver-minor>=0.2.2-patch2.post3+buildmeta,<0.3.0",
                            "semver-patch>=0.0.2+bm,<0.0.3",
                        ],
                    ),
                ),
                ("dev", ("requirements for dev", ["buz>=1.3.0,<2.0.0", "oof==0.3.4", "rab"])),
                (
                    "all-prod",
                    (
                        "Combined dependencies for all TVM pieces, excluding dev",
                        [
                            "bar==1.0",
                            "baz>2.3",
                            "buz>=1.3.0,<2.0.0",
                            "foo",
                            "non-constrained",
                            "qux~=1.2.4",
                            "semver-minor>=0.2.2-patch2.post3+buildmeta,<0.3.0",
                            "semver-patch>=0.0.2+bm,<0.0.3",
                        ],
                    ),
                ),
            ]
        )


def test_semver():
    problems = []

    assert gen_requirements.parse_semver("C", "^1.2.0", problems) == (["1", "2", "0"], 0, 1)
    assert problems == []

    assert gen_requirements.parse_semver("C", "^0.2.0", problems) == (["0", "2", "0"], 1, 2)
    assert problems == []

    assert gen_requirements.parse_semver("C", "^0.0.0", problems) == (["0", "0", "0"], 0, 0)
    assert problems == []

    assert gen_requirements.parse_semver("C", "^0.a.0", problems) == ([], 0, 0)
    assert problems == ["C: invalid semver constraint ^0.a.0"]


if __name__ == "__main__":
    tvm.testing.main()
