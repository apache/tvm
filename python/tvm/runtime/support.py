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

"""Runtime support infra of TVM."""

import re
from typing import TypeVar

import tvm.ffi


@tvm.ffi.register_func("tvm.runtime.regex_match")
def _regex_match(regex_pattern: str, match_against: str) -> bool:
    """Check if a pattern matches a regular expression

    This function should be used instead of `std::regex` within C++
    call sites, to avoid ABI incompatibilities with pytorch.

    Currently, the pytorch wheels available through pip install use
    the pre-C++11 ABI by setting `-DUSE_CXX11_ABI=0` [0]. If TVM were to
    user the pre-C++11 ABI, this would cause breakages with
    dynamically-linked LLVM environments.

    Use of the `<regex>` header in TVM should be avoided, as its
    implementation is not supported by gcc's dual ABI. This ABI
    incompatibility results in runtime errors either when `std::regex`
    is called from TVM, or when `std::regex` is called from pytorch,
    depending on which library was loaded first.  This restriction can
    be removed when a version of pytorch compiled using
    `-DUSE_CXX11_ABI=1` is available from PyPI.

    This is exposed as part of `libtvm_runtime.so` as it is used by
    the DNNL runtime.

    [0] https://github.com/pytorch/pytorch/issues/51039

    Parameters
    ----------
    regex_pattern: str

         The regular expression

    match_against: str

        The string against which to match the regular expression

    Returns
    -------
    match_result: bool

        True if `match_against` matches the pattern defined by
        `regex_pattern`, and False otherwise.

    """
    match = re.match(regex_pattern, match_against)
    return match is not None


T = TypeVar("T")


def derived_object(cls: type[T]) -> type[T]:
    """A decorator to register derived subclasses for TVM objects.

    Parameters
    ----------
    cls : type
        The derived class to be registered.

    Returns
    -------
    cls : type
        The decorated TVM object.

    Example
    -------
    .. code-block:: python

        @register_object("meta_schedule.PyRunner")
        class _PyRunner(meta_schedule.Runner):
            def __init__(self, f_run: Callable = None):
                self.__init_handle_by_constructor__(_ffi_api.RunnerPyRunner, f_run)

        class PyRunner:
            _tvm_metadata = {
                "cls": _PyRunner,
                "methods": ["run"]
            }
            def run(self, runner_inputs):
                raise NotImplementedError

        @derived_object
        class LocalRunner(PyRunner):
            def run(self, runner_inputs):
                ...
    """

    import functools  # pylint: disable=import-outside-toplevel
    import weakref  # pylint: disable=import-outside-toplevel

    def _extract(inst: type, name: str):
        """Extract function from intrinsic class."""

        def method(*args, **kwargs):
            return getattr(inst, name)(*args, **kwargs)

        for inherit_cls, base_cls in zip(cls.__mro__, cls.__mro__[1:]):
            # extract functions that differ from the base class
            if not hasattr(base_cls, name):
                continue
            if getattr(base_cls, name) is getattr(inherit_cls, name) and name != "__str__":
                continue
            return method

        # for task scheduler return None means calling default function
        # otherwise it will trigger a TVMError of method not implemented
        # on the c++ side when you call the method, __str__ not required
        return None

    assert isinstance(cls.__base__, type)
    if hasattr(cls, "_type") and cls._type == "TVMDerivedObject":  # type: ignore
        raise TypeError(
            (
                f"Inheritance from a decorated object `{cls.__name__}` is not allowed. "
                f"Please inherit from `{cls.__name__}._cls`."
            )
        )
    assert hasattr(
        cls, "_tvm_metadata"
    ), "Please use the user-facing method overriding class, i.e., PyRunner."

    base = cls.__base__
    metadata = getattr(base, "_tvm_metadata")
    fields = metadata.get("fields", [])
    methods = metadata.get("methods", [])

    class TVMDerivedObject(metadata["cls"]):  # type: ignore
        """The derived object to avoid cyclic dependency."""

        _cls = cls
        _type = "TVMDerivedObject"

        def __init__(self, *args, **kwargs):
            """Constructor."""
            self._inst = cls(*args, **kwargs)

            super().__init__(
                # the constructor's parameters, builder, runner, etc.
                *[getattr(self._inst, name) for name in fields],
                # the function methods, init_with_tune_context, build, run, etc.
                *[_extract(self._inst, name) for name in methods],
            )

            # for task scheduler hybrid funcs in c++ & python side
            # using weakref to avoid cyclic dependency
            self._inst._outer = weakref.ref(self)

        def __getattr__(self, name):
            import inspect  # pylint: disable=import-outside-toplevel

            try:
                # fall back to instance attribute if there is not any
                # return self._inst.__getattribute__(name)
                result = self._inst.__getattribute__(name)
            except AttributeError:
                result = super(TVMDerivedObject, self).__getattr__(name)

            if inspect.ismethod(result):

                def method(*args, **kwargs):
                    return result(*args, **kwargs)

                # set __own__ to aviod implicit deconstruction
                setattr(method, "__own__", self)
                return method

            return result

        def __setattr__(self, name, value):
            if name not in ["_inst", "key", "handle"]:
                self._inst.__setattr__(name, value)
            else:
                super(TVMDerivedObject, self).__setattr__(name, value)

    functools.update_wrapper(TVMDerivedObject.__init__, cls.__init__)  # type: ignore
    TVMDerivedObject.__name__ = cls.__name__
    TVMDerivedObject.__doc__ = cls.__doc__
    TVMDerivedObject.__module__ = cls.__module__
    for key, value in cls.__dict__.items():
        if isinstance(value, (classmethod, staticmethod)):
            setattr(TVMDerivedObject, key, value)
    return TVMDerivedObject
