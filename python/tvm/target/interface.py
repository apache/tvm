from collections import namedtuple
from dataclasses import dataclass, field
from functools import wraps, partial
import inspect
import itertools
from tvm.tir.function import PrimFunc
from tvm import tir
from typing import Dict, Iterable, List

Resource = namedtuple("Resource", ["name", "count"])


@dataclass(frozen=True)
class IntrinsicDeclaration:
    """A Dataclass representing an intrinsic declaration.

    This is meant to act as a representative object for the declared intrinsic
    and bundles the name, description, and implementation. This dataclass can
    be extended with additional metadata as needed.
    """

    desc: PrimFunc = field(repr=False)
    impl: PrimFunc = field(repr=False)
    name: str
    # Example metadata representing async capability/resource use
    consumes: List[Resource] = field(default_factory=list)


def consumes(resource_name, count=1):
    """
    An example metadata decorator which annotates an intrinsic declaration with
    a resource consumption tag. If the annotated intrinsic does not already
    have a `consumes` attribute, this decorator will first add it.

    Examples
    --------
    The following stub shows using the decorator in conjunction with
    `intrinsic` to declare a new intrinsic with the given resource
    consumption metadata

    .. code-block:: python

        @consumes('example resource')
        @intrinsic
        class MyFunction:
            ...
    """

    def decorator(func):
        if hasattr(func, "consumes") and isinstance(func.consumes, list):
            func.consumes.append(Resource(resource_name, count))
        elif hasattr(func, "consumes"):
            raise TypeError(
                "Consumes decorator expects the consumes attribute to be a "
                f"list, but got {type(func.consumes)}"
            )
        else:
            func.consumes = list()
            func.consumes.append(Resource(resource_name, count))
        return func

    return decorator


def intrinsic(func=None, *, name=None, desc_name="desc", impl_name="impl"):
    """A decorator for declaring an intrinsic function.

    This decorator unifies the intrinsic description and implementation while
    handling the Tensor registration call. There are two possible approaches to
    using this decorator as it can be attached to a function definition or a
    class. In general, the class syntax is recommended over the function
    syntax.

    The decorator replaces the attached function or class with an
    `IntrinsicDeclaration` object

    Parameters
    ----------
    name : str, optional
        The name of the intrinsic, will be used to register the intrinsic. If
        not provided, the name of the attached function/class will be used.

    desc_name : str, optional
        The name of the description function. Used when attached to a class.

    impl_name : str, optional
        The name of the implementation function. Used when attached to a class.


    Examples
    --------
    The following example shows the two methods of using this decorator to
    declare the same intrinsic.

    .. code-block:: python

        @intrinsic
        class MyIntrin:
            @T.prim_func
            def desc(...): ...

            @T.prim_func
            def impl(...): ...

        @intrinsic
        def MyIntrinAlt():
            @T.prim_func
            def desc(...): ...

            @T.prim_func
            def impl(...): ...

            return desc, impl

        # tensorize via the attached name
        s.tensorize(i, MyIntrin.name)    # "MyIntrin"
        s.tensorize(i, MyIntrinAlt.name) # "MyIntrinAlt"

    See Also
    --------
    IntrinsicDeclaration
    IntrinsicInterface.intrinsic
    """

    # this is based on recipe 9.6 from python cookbook and is here to allow the
    # decorator to be used both with and without the call syntax
    if func is None:
        return partial(intrinsic, name=name, desc_name=desc_name, impl_name=impl_name)

    name = name or func.__name__

    desc, impl = None, None

    # if this is attached to a class, pull the appropriate attributes
    if isinstance(func, type):
        desc, impl = getattr(func, desc_name), getattr(func, impl_name)

    # otherwise assume we are attached to a function and call it, the function
    # must return desc, impl in that exact order
    else:
        desc, impl = func()

    tir.TensorIntrin.register(name, desc, impl)
    inner = IntrinsicDeclaration(desc, impl, name)

    if hasattr(func, "consumes"):
        inner.consumes.extend(func.consumes)

    return inner


class IntrinsicInterface:
    registry: Dict[str, IntrinsicDeclaration]
    name: str
    resources: Dict[str, int]

    def __init__(self, name):
        self.registry = dict()
        self.name = name
        self.resources = dict()

    def intrinsic(self, func=None, *, name=None, name_prefix=True, **kwargs):

        if func is None:
            return partial(
                self.intrinsic,
                name=name,
                name_prefix=name_prefix,
                **kwargs,
            )

        # this abomination is a tricky means of providing automatic name
        # mangling when running inside a function annotated with
        # `run_generator`
        frame_stack = inspect.stack()
        if (
            not name
            and len(frame_stack) > 2
            and frame_stack[2].function == "__call__"
            and "self" in frame_stack[2].frame.f_locals
            and isinstance(frame_stack[2].frame.f_locals["self"], GeneratorWrapper)
        ):
            args = frame_stack[2].frame.f_locals["args"]
            kwargs = frame_stack[2].frame.f_locals["kwargs"]
            if args or kwargs:
                name = (
                    func.__name__
                    + "_"
                    + "_".join((str(arg) for arg in args))
                    + "_".join((str(val) for val in kwargs.values()))
                )

        name = name or f"{self.name}_{func.__name__}"

        if not name.startswith(self.name) and name_prefix:
            name = f"{self.name}_{name}"

        assert name not in self.registry, (
            f"An intrinsic named {name} has already been registered, a second"
            "intrinsic cannot be registered under the same name. If using a "
            "generator, please ensure that the generator produces a uniquely "
            "named intrinsic per each unique input set"
        )

        inner = intrinsic(func, name=name, **kwargs)
        self.registry[name] = inner

        return inner

    def set_resource(self, name: str, count: int):
        self.resources[name] = count

    def set_resources_from_dict(self, resource_dict: dict):
        self.resources.update(
            (
                (key, value)
                for key, value in resource_dict.items()
                if isinstance(key, str) and isinstance(value, int)
            )
        )

    @staticmethod
    def create_interface(cls):
        """Utility decorator used to turn a class declaration into an interface
        object.

        It's probably better to just make the object directly by
        initializing an IntrinsicInterface since this can confuse IDEs.
        Currently only handles resource declarations but could be extended in
        the future as needed.

        Examples
        --------
        .. code-block:: python
            @IntrinsicInterface.create_interface
            class SecondInterface:
                resources = {"test_resource": 1}

        """

        inner = IntrinsicInterface(cls.__name__)
        if hasattr(cls, "resources"):
            # the validation code in these if-arms is inefficient but given
            # that these resource dicts are unlikely to be all that large this
            # is fine for the time being. If the need arises these can be
            # converted to a more efficient but less readable single pass
            # version

            if isinstance(cls.resources, dict) and all(
                (
                    isinstance(key, str) and isinstance(value, int)
                    for key, value in cls.resources.items()
                )
            ):
                inner.resources = cls.resources
            elif isinstance(cls.resources, Iterable) and all(
                (isinstance(item, (Resource, tuple)) and len(item) == 2 for item in cls.resources)
            ):
                inner.resources = {key: value for key, value in cls.resources}
                cls.resources = inner.resources
            else:
                raise TypeError(
                    "the resources field must be a dictionary with string keys"
                    " and int values or an Iterable of valid Resource tuples"
                )
        else:
            cls.resources = inner.resources

        class WrappedIntrinsicInterface(cls):
            _inner = inner
            registry = _inner.registry
            function = _inner.intrinsic

        return WrappedIntrinsicInterface


class GeneratorWrapper:
    """
    A wrapper class for functions which generate intrinsic implementations.
    Used to handle automatic name mangling based on generator input
    """

    def __init__(self, wrapped_fn):
        self._wrapped_fn = wrapped_fn
        self.captured_output = []

    def __call__(self, *args, **kwargs):
        self._wrapped_fn(*args, **kwargs)


def generator(func):
    """
    Decorator for wrapping functions which generate intrinsic implementations
    """
    gen = GeneratorWrapper(func)
    # this is needed to ensure that signature introspection works properly
    gen = wraps(func)(gen)
    return gen


def run_generator(*, _validators=None, **kwargs):
    """
    This decorator when given iterable keyword arguments will run the annotated
    generator with the cartesian product of the input iterators. Optional
    validators may be supplied via the `_validators` argument. If present, this
    decorator will skip over any argument lists which fail to pass all
    validators.

    Successful non-None values returned from the attached function will be
    captured in the `captured_output` field of the returned function wrapper.

    Note: wraps the generator function with the :code:`@generator` wrapper if
    it has not already been applied
    """

    if _validators:
        if not isinstance(_validators, Iterable):
            _validators = [_validators]
    else:
        _validators = []

    product = itertools.product(*kwargs.values())

    def decorator(func):
        if not isinstance(func, GeneratorWrapper):
            func = generator(func)

        signature = inspect.signature(func)
        arg_ordering = [
            key
            for key, value in signature.parameters.items()
            if value.kind == value.POSITIONAL_ONLY
        ]

        for concrete_values in product:
            arg_dict = {key: value for key, value in zip(kwargs.keys(), concrete_values)}

            args = [arg_dict[key] for key in arg_ordering]
            arg_dict = {key: value for key, value in arg_dict.items() if key not in arg_ordering}
            bound = signature.bind(*args, **arg_dict)
            bound.apply_defaults()

            passed_validation = True
            for validator in _validators:
                validator_signature = inspect.signature(validator)

                val_args = []
                val_kwargs = {}

                for param_name, info in validator_signature.parameters.items():
                    if info.kind in {
                        info.POSITIONAL_ONLY,
                        info.POSITIONAL_OR_KEYWORD,
                    }:
                        val_args.append(bound.arguments[param_name])
                    elif info.kind == info.KEYWORD_ONLY:
                        val_kwargs[param_name] = bound.arguments[param_name]
                    else:
                        raise ValueError(
                            "validators may not contain variable positional or" " keyword arguments"
                        )

                result = validator(*val_args, **val_kwargs)
                if not result:
                    passed_validation = False
                    break

            if passed_validation:
                output = func(*bound.args, **bound.kwargs)
                if output is not None:
                    func.captured_output.append(output)

        return func

    return decorator


# friendly alias
create_interface = IntrinsicInterface.create_interface
