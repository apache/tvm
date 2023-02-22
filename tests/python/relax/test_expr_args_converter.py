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

from typing import Any, Callable, List, Optional, Union

import pytest
import tvm
import tvm.testing
from tvm import relax
from tvm.relax import Expr
from tvm.relax.utils import args_converter


def _test_base(f_checker: Callable, arg: Any, *args: Any, **kwargs: Any) -> None:
    # Test converting to `Expr`
    assert f_checker(arg)
    # Test converting `*args`
    assert isinstance(args, tuple)
    assert all([f_checker(arg) for arg in args])
    # Test converting `**kwargs`
    assert isinstance(kwargs, dict)
    assert all([f_checker(arg) for arg in kwargs.values()])


def _test_expr(arg: Expr, *args: Expr, **kwargs: Expr) -> None:
    f_checker = lambda x: isinstance(x, Expr)
    _test_base(f_checker, arg, *args, **kwargs)


def _test_optional_expr(
    arg: Optional[Expr], *args: Optional[Expr], **kwargs: Optional[Expr]
) -> None:
    f_checker = lambda x: x is None or isinstance(x, Expr)
    _test_base(f_checker, arg, *args, **kwargs)


def _test_list_expr(arg: List[Expr], *args: List[Expr], **kwargs: List[Expr]) -> None:
    f_checker = lambda x: isinstance(x, list) and all([isinstance(arg, Expr) for arg in x])
    _test_base(f_checker, arg, *args, **kwargs)


def _test_optional_list_expr(
    arg: Optional[List[Expr]], *args: Optional[List[Expr]], **kwargs: Optional[List[Expr]]
) -> None:
    f_checker = lambda x: x is None or (
        isinstance(x, list) and all([isinstance(arg, Expr) for arg in x])
    )
    _test_base(f_checker, arg, *args, **kwargs)


prim_value = 1
str_value = "value_to_be_convert"
shape_value = (1, 1)
tuple_value = (relax.const(1), (1, 1))
placeholder = relax.const(0)

test_cases = [prim_value, str_value, shape_value, tuple_value, placeholder]


def test_args_to_expr():
    for _f in [_test_expr, _test_optional_expr]:
        f = args_converter.to_expr("arg", "args", "kwargs")(_f)
        for x in test_cases:
            f(
                x,
                x,  # the first argument in *args
                x,  # the second argument in *args
                test_kwargs=x,
            )

            if _f == _test_optional_expr:
                f(None, None, x, test_kwargs=None)


def test_args_to_list_expr():
    for _f in [_test_list_expr, _test_optional_list_expr]:
        f = args_converter.to_list_expr("arg", "args", "kwargs")(_f)
        for x in test_cases:
            f(
                [x],
                [x],  # the first argument in *args
                [x, x],  # the second argument in *args
                test_kwargs=[x, (x,)],
            )

            if _f == _test_optional_list_expr:
                f(None, None, [x], test_kwargs=None)


def test_error():
    f = args_converter.to_list_expr("arg", "args", "kwargs")(_test_list_expr)
    with pytest.raises(TypeError):
        f(prim_value)  # fail to convert prim_value to `List[Expr]`


def test_auto_convert():
    for _f in [_test_expr, _test_optional_expr]:
        f = args_converter.auto(_f)
        for x in test_cases:
            f(x, (x,), test_kwargs=x)

            if _f == _test_optional_expr:
                f(None, x, test_kwargs=None)

    for _f in [_test_list_expr, _test_optional_list_expr]:
        f = args_converter.auto(_f)
        for x in test_cases:
            f([x], [x, x], test_kwargs=[x, (x,)])

            if _f == _test_optional_list_expr:
                f(None, None, [x], test_kwargs=None)


def test_auto_convert_skip():
    def _test_expr_skip(arg: int, *args: Union[str, Expr], **kwargs: List[Optional[Expr]]) -> None:
        f_checker = lambda x: not isinstance(x, Expr)
        _test_base(f_checker, arg, *args, **kwargs)

    f = args_converter.auto(_test_expr_skip)
    f(1, "str", test_kwargs=[None])


def test_empty_tuple():
    def _test(arg: Expr):
        assert isinstance(arg, relax.Tuple)

    f = args_converter.auto(_test)
    f(())


if __name__ == "__main__":
    tvm.testing.main()
