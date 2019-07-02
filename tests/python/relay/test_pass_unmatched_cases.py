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

import tvm
from tvm import relay
from tvm.relay.prelude import Prelude
from tvm.relay.analysis import unmatched_cases

def test_empty_match_block():
    # empty match block will not match anything, so it should return a wildcard pattern
    v = relay.Var('v')
    match = relay.Match(v, [])

    unmatched = unmatched_cases(match)
    assert len(unmatched) == 1
    assert isinstance(unmatched[0], relay.PatternWildcard)


def test_trivial_matches():
    # a match clause with a wildcard will match anything
    v = relay.Var('v')
    match = relay.Match(v, [
        relay.Clause(relay.PatternWildcard(), v)
    ])
    assert len(unmatched_cases(match)) == 0

    # same with a pattern var
    w = relay.Var('w')
    match = relay.Match(v, [
        relay.Clause(relay.PatternVar(w), w)
    ])
    assert len(unmatched_cases(match)) == 0


def test_single_constructor_adt():
    mod = relay.Module()
    box = relay.GlobalTypeVar('box')
    a = relay.TypeVar('a')
    box_ctor = relay.Constructor('box', [a], box)
    box_data = relay.TypeData(box, [a], [box_ctor])
    mod[box] = box_data

    v = relay.Var('v')
    match = relay.Match(v, [
        relay.Clause(relay.PatternConstructor(box_ctor, [relay.PatternWildcard()]), v)
    ])

    # with one constructor, having one pattern constructor case is exhaustive
    assert len(unmatched_cases(match, mod)) == 0

    # this will be so if we nest the constructors too
    nested_pattern = relay.Match(v, [
        relay.Clause(
            relay.PatternConstructor(
                box_ctor,
                [relay.PatternConstructor(box_ctor,
                                          [relay.PatternConstructor(
                                              box_ctor,
                                              [relay.PatternWildcard()])])]), v)
    ])
    assert len(unmatched_cases(nested_pattern, mod)) == 0


def test_too_specific_match():
    mod = relay.Module()
    p = Prelude(mod)

    v = relay.Var('v')
    match = relay.Match(v, [
        relay.Clause(
            relay.PatternConstructor(
                p.cons, [relay.PatternWildcard(),
                         relay.PatternConstructor(p.cons, [relay.PatternWildcard(),
                                                           relay.PatternWildcard()])]), v)
    ])

    unmatched = unmatched_cases(match, mod)

    # will not match nil or a list of length 1
    nil_found = False
    single_length_found = False
    assert len(unmatched) == 2
    for case in unmatched:
        assert isinstance(case, relay.PatternConstructor)
        if case.constructor == p.nil:
            nil_found = True
        if case.constructor == p.cons:
            assert isinstance(case.patterns[1], relay.PatternConstructor)
            assert case.patterns[1].constructor == p.nil
            single_length_found = True
    assert nil_found and single_length_found

    # if we add a wildcard, this should work
    new_match = relay.Match(v, [
        relay.Clause(
            relay.PatternConstructor(
                p.cons, [relay.PatternWildcard(),
                         relay.PatternConstructor(p.cons, [relay.PatternWildcard(),
                                                           relay.PatternWildcard()])]), v),
        relay.Clause(relay.PatternWildcard(), v)
    ])
    assert len(unmatched_cases(new_match, mod)) == 0


def test_multiple_constructor_clauses():
    mod = relay.Module()
    p = Prelude(mod)

    v = relay.Var('v')
    match = relay.Match(v, [
        # list of length exactly 1
        relay.Clause(
            relay.PatternConstructor(p.cons, [relay.PatternWildcard(),
                                              relay.PatternConstructor(p.nil, [])]), v),
        # list of length exactly 2
        relay.Clause(
            relay.PatternConstructor(
                p.cons, [relay.PatternWildcard(),
                         relay.PatternConstructor(p.cons, [relay.PatternWildcard(),
                                                           relay.PatternConstructor(p.nil, [])
                         ])]), v),
        # empty list
        relay.Clause(
            relay.PatternConstructor(p.nil, []), v),
        # list of length 2 or more
        relay.Clause(
            relay.PatternConstructor(
                p.cons, [relay.PatternWildcard(),
                         relay.PatternConstructor(p.cons, [relay.PatternWildcard(),
                                                           relay.PatternWildcard()])]), v)
    ])
    assert len(unmatched_cases(match, mod)) == 0


def test_missing_in_the_middle():
    mod = relay.Module()
    p = Prelude(mod)

    v = relay.Var('v')
    match = relay.Match(v, [
        # list of length exactly 1
        relay.Clause(
            relay.PatternConstructor(p.cons, [relay.PatternWildcard(),
                                              relay.PatternConstructor(p.nil, [])]), v),
        # empty list
        relay.Clause(
            relay.PatternConstructor(p.nil, []), v),
        # list of length 3 or more
        relay.Clause(
            relay.PatternConstructor(
                p.cons, [relay.PatternWildcard(),
                         relay.PatternConstructor(
                             p.cons,
                             [relay.PatternWildcard(),
                              relay.PatternConstructor(
                                  p.cons,
                                  [relay.PatternWildcard(),
                                   relay.PatternWildcard()])])]),
            v)
    ])

    # fails to match a list of length exactly two
    unmatched = unmatched_cases(match, mod)
    assert len(unmatched) == 1
    assert isinstance(unmatched[0], relay.PatternConstructor)
    assert unmatched[0].constructor == p.cons
    assert isinstance(unmatched[0].patterns[1], relay.PatternConstructor)
    assert unmatched[0].patterns[1].constructor == p.cons
    assert isinstance(unmatched[0].patterns[1].patterns[1], relay.PatternConstructor)
    assert unmatched[0].patterns[1].patterns[1].constructor == p.nil


def test_mixed_adt_constructors():
    mod = relay.Module()
    box = relay.GlobalTypeVar('box')
    a = relay.TypeVar('a')
    box_ctor = relay.Constructor('box', [a], box)
    box_data = relay.TypeData(box, [a], [box_ctor])
    mod[box] = box_data

    p = Prelude(mod)

    v = relay.Var('v')
    box_of_lists_inc = relay.Match(v, [
        relay.Clause(
            relay.PatternConstructor(
                box_ctor,
                [relay.PatternConstructor(p.cons, [
                    relay.PatternWildcard(), relay.PatternWildcard()])]), v)
    ])

    # will fail to match a box containing an empty list
    unmatched = unmatched_cases(box_of_lists_inc, mod)
    assert len(unmatched) == 1
    assert isinstance(unmatched[0], relay.PatternConstructor)
    assert unmatched[0].constructor == box_ctor
    assert len(unmatched[0].patterns) == 1 and unmatched[0].patterns[0].constructor == p.nil

    box_of_lists_comp = relay.Match(v, [
        relay.Clause(
            relay.PatternConstructor(
                box_ctor, [relay.PatternConstructor(p.nil, [])]), v),
        relay.Clause(
            relay.PatternConstructor(
                box_ctor, [relay.PatternConstructor(p.cons, [
                    relay.PatternWildcard(), relay.PatternWildcard()])]), v)
    ])
    assert len(unmatched_cases(box_of_lists_comp, mod)) == 0

    list_of_boxes_inc = relay.Match(v, [
        relay.Clause(
            relay.PatternConstructor(
                p.cons, [relay.PatternConstructor(box_ctor, [relay.PatternWildcard()]),
                         relay.PatternWildcard()]), v)
    ])

    # fails to match empty list of boxes
    unmatched = unmatched_cases(list_of_boxes_inc, mod)
    assert len(unmatched) == 1
    assert isinstance(unmatched[0], relay.PatternConstructor)
    assert unmatched[0].constructor == p.nil

    list_of_boxes_comp = relay.Match(v, [
        # exactly one box
        relay.Clause(
            relay.PatternConstructor(
                p.cons, [relay.PatternConstructor(box_ctor, [relay.PatternWildcard()]),
                         relay.PatternConstructor(p.nil, [])]), v),
        # exactly two boxes
        relay.Clause(
            relay.PatternConstructor(
                p.cons, [relay.PatternConstructor(box_ctor, [relay.PatternWildcard()]),
                         relay.PatternConstructor(p.cons, [
                             relay.PatternConstructor(box_ctor, [relay.PatternWildcard()]),
                             relay.PatternConstructor(p.nil, [])
                         ])]), v),
        # exactly three boxes
        relay.Clause(
            relay.PatternConstructor(
                p.cons, [relay.PatternConstructor(box_ctor, [relay.PatternWildcard()]),
                         relay.PatternConstructor(p.cons, [
                             relay.PatternConstructor(box_ctor, [relay.PatternWildcard()]),
                             relay.PatternConstructor(p.cons, [
                                 relay.PatternConstructor(box_ctor, [relay.PatternWildcard()]),
                                 relay.PatternConstructor(p.nil, [])
                             ])])]), v),
        # one or more boxes
        relay.Clause(relay.PatternConstructor(p.cons, [relay.PatternWildcard(),
                                                       relay.PatternWildcard()]), v),
        # no boxes
        relay.Clause(relay.PatternConstructor(p.nil, []), v)
    ])
    assert len(unmatched_cases(list_of_boxes_comp, mod)) == 0
