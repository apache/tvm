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
"""Function data types."""

from typing import Callable, List, Mapping, Optional, Union, Tuple
import inspect

import tvm
import tvm._ffi
import tvm.runtime
from tvm.runtime import Object
from tvm.ir import BaseFunc, Range
from .buffer import Buffer
from .expr import Var, PrimExpr
from . import _ffi_api


@tvm._ffi.register_object("tir.PrimFunc")
class PrimFunc(BaseFunc):
    """A function declaration expression.

    Parameters
    ----------
    params: List[Union[tvm.tir.Var, tvm.tir.Buffer]]
        List of input parameters to the function.

    body: tvm.tir.Stmt
        The body of the function.

    ret_type: tvm.ir.Type
        The return type annotation of the function.

    buffer_map : Map[tvm.tir.Var, tvm.tir.Buffer]
        The buffer binding map.

    preflattened_buffer_map : Optional[Map[tvm.tir.Var, tvm.tir.Buffer]]
        The buffer binding map, prior to any flattening.

    attrs: Optional[tvm.Attrs]
        Attributes of the function, can be None

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(
        self,
        params,
        body,
        ret_type=None,
        buffer_map=None,
        preflattened_buffer_map=None,
        attrs=None,
        span=None,
    ):

        param_list = []
        buffer_map = {} if buffer_map is None else buffer_map
        preflattened_buffer_map = {} if preflattened_buffer_map is None else preflattened_buffer_map
        for x in params:
            x = tvm.runtime.convert(x) if not isinstance(x, Object) else x
            if isinstance(x, Buffer):
                var = Var(x.name, dtype="handle")
                param_list.append(var)
                buffer_map[var] = x
            elif isinstance(x, Var):
                param_list.append(x)
            else:
                raise TypeError("params can only contain Var or Buffer")

        self.__init_handle_by_constructor__(
            _ffi_api.PrimFunc,
            param_list,
            body,
            ret_type,
            buffer_map,
            preflattened_buffer_map,
            attrs,
            span,
        )  # type: ignore

    def with_body(self, new_body, span=None):
        """Create a new PrimFunc with the same set signatures but a new body.

        Parameters
        ----------
        new_body : Stmt
            The new body.

        span : Optional[Span]
            The location of this itervar in the source code.

        Returns
        -------
        new_func : PrimFunc
            The created new function.
        """
        return PrimFunc(
            self.params,
            new_body,
            self.ret_type,
            self.buffer_map,
            self.preflattened_buffer_map,
            self.attrs,
            span,
        )

    def specialize(self, param_map: Mapping[Var, Union[PrimExpr, Buffer]]):
        """Specialize parameters of PrimFunc

        Parameters
        ----------

        param_map : Mapping[Var, Union[PrimExpr, Buffer]]
            The mapping from function params to the instance

        Examples
        --------
        We can define a Meta TIR function with symbolic shape:

        .. code-block:: python

            @T.prim_func
            def mem_copy(a: T.handle, b: T.handle, m: T.int32, n: T.int32) -> None:
                A = T.match_buffer(a, (m, n), "float32")
                B = T.match_buffer(b, (m, n), "float32")

                for i, j in T.grid(m, n):
                    with T.block():
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj]

        Then we can make it specialized with given shapes or buffers.

        .. code-block:: python

            a, _, m, n = mem_copy.params
            func = mem_copy.specialize({a: tir.decl_buffer((16, 16))})
            # or
            func = mem_copy.specialize({n: 16, m: 16})

        The specialized function:

        .. code-block:: python

            @T.prim_func
            def mem_copy_16_16(a: T.handle, b: T.handle) -> None:
                A = T.match_buffer(a, (16, 16), "float32")
                B = T.match_buffer(b, (16, 16), "float32")

                for i, j in T.grid(16, 16):
                    with T.block():
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj]

        Returns
        -------
        func : PrimFunc
            The new function with parameter specialized
        """
        return _ffi_api.Specialize(self, param_map)  # type: ignore

    def script(self, tir_prefix: str = "T", show_meta: bool = False) -> str:
        """Print IRModule into TVMScript

        Parameters
        ----------
        tir_prefix : str
            The tir namespace prefix

        show_meta : bool
            Whether to show meta information

        Returns
        -------
        script : str
            The TVM Script of the PrimFunc
        """
        return tvm._ffi.get_global_func("script.AsTVMScript")(
            self, tir_prefix, show_meta
        )  # type: ignore


@tvm._ffi.register_object("tir.TensorIntrin")
class TensorIntrin(Object):
    """A tensor intrinsic.

    Parameters
    ----------
    desc : PrimFunc
        The function to describe the computation.

    impl : PrimFunc
        The function of the implementation for the execution.
    """

    def __init__(self, desc, impl):
        self.__init_handle_by_constructor__(_ffi_api.TensorIntrin, desc, impl)

    @staticmethod
    def register(name: str, desc: PrimFunc, impl: PrimFunc):
        """Register a tensor intrinsic with its name.

        Parameters
        ----------
        name : str
            The name of the TensorIntrin to register.
        desc : PrimFunc
            The function to describe the computation.
        impl : PrimFunc
            The function of the implementation for the execution.
        """
        return _ffi_api.TensorIntrinRegister(name, TensorIntrin(desc, impl))  # type: ignore

    @staticmethod
    def get(name: str):
        """Look up a tensor intrinsic by its name.

        Parameters
        ----------
        name : str
            The name of the TensorIntrin to look up.

        Returns
        -------
        result : TensorIntrin
            The TensorIntrin with the specified name.
        """
        return _ffi_api.TensorIntrinGet(name)  # pylint: type: ignore


@tvm._ffi.register_object("tir.IndexMap")
class IndexMap(Object):
    """A mapping from multi-dimensional indices to another set of multi-dimensional indices

    Parameters
    ----------
    initial_indices : List[Var]
        Variables representing the indices prior to remapping.
    final_indices : List[PrimExpr]
        Expressions defining the indices after remapping.
    """

    initial_indices: List[Var]
    final_indices: List[PrimExpr]

    def __init__(self, initial_indices, final_indices):
        self.__init_handle_by_constructor__(_ffi_api.IndexMap, initial_indices, final_indices)

    @staticmethod
    def from_func(mapping_function: Callable, ndim: Optional[int] = None):
        """Create an index map from a function

        Parameters
        ----------
        mapping_function : Callable
            The function to map from source indices to target indices
        """
        params = inspect.signature(mapping_function).parameters
        default_index_dtype = "int32"
        args = []
        var_arg_name = None
        for name, param in params.items():
            if param.kind in [
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ]:
                args.append(tvm.tir.Var(name, default_index_dtype))
            elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                var_arg_name = name
            else:
                raise ValueError("transform_layout mapping may not have *args or **kwargs")

        # Now that all the named arguments have been collected,
        # everything that remains should go to the *args, if
        # specified.
        if var_arg_name is not None:
            assert ndim is not None, "ndim must be specified when *args is used"
            num_var_args = ndim - len(args)
            for i in range(num_var_args):
                args.append(tvm.tir.Var(f"{var_arg_name}_{i}", default_index_dtype))

        final_indices = mapping_function(*args)
        return IndexMap(args, final_indices)

    def is_equivalent_to(self, other_map: "IndexMap") -> bool:
        """Return if the index maps are equivalent.

        Parameters
        ----------
        other_map: IndexMap

            The IndexMap to which the comparison should be made.

        Returns
        -------
        is_equivalent: bool

            True if the two mappings represent the same
            transformation, otherwise False
        """
        if len(self.initial_indices) != len(other_map.initial_indices):
            return False
        if len(self.final_indices) != len(other_map.final_indices):
            return False

        analyzer = tvm.arith.Analyzer()

        mapped_other_final_indices = other_map.map_indices(self.initial_indices)
        for self_index, other_index in zip(self.final_indices, mapped_other_final_indices):
            if not analyzer.can_prove_equal(self_index, other_index):
                return False

        return True

    def map_indices(self, indices: List[PrimExpr]) -> List[PrimExpr]:
        """Apply the index map to a set of indices

        Parameters
        ----------
        indices : List[PrimExpr]
            The indices to be mapped

        Returns
        -------
        result : List[PrimExpr]
            The mapped indices
        """
        return _ffi_api.IndexMapMapIndices(self, indices)

    def map_shape(self, shape: List[PrimExpr]) -> List[PrimExpr]:
        """Apply the index map to a buffer shape

        Parameters
        ----------
        shape : List[PrimExpr]
            The buffer shape to be mapped

        Returns
        -------
        result : List[PrimExpr]
            The mapped shape
        """
        return _ffi_api.IndexMapMapShape(self, shape)

    def inverse(self, shape: List[Union[Range, PrimExpr]]) -> "IndexMap":
        """Return the inverse of the map

        Throws an error if the function is not bijective.

        Parameters
        ----------
        shape: List[Union[Range,PrimExpr]]

            The region over which the inverse should be determined.
            Used for validating that the mapping is bijective over
            this range.

        Returns
        -------
        inverse : IndexMap

            The inverse
        """

        shape = [dim if isinstance(dim, Range) else Range(0, dim) for dim in shape]
        return _ffi_api.IndexMapInverse(self, shape)

    def non_surjective_inverse(
        self, shape: List[Union[Range, PrimExpr]]
    ) -> Tuple["IndexMap", PrimExpr]:
        """Return the inverse of the map

        Can be applied to transformations that introduce padding.

        Parameters
        ----------
        shape: List[Union[Range,PrimExpr]]

            The region over which the inverse should be determined.
            Used for determining the predicate.

        Returns
        -------
        result : Tuple[IndexMap, PrimExpr]

            The inverse, and a predicate for which the inverse maps to
            a valid index in the input range.

        Examples
        --------

        .. code-block:: python

            index_map = IndexMap.from_func(lambda i: [i//4, i%4])
            inverse_map, predicate = index_map.non_surjective_inverse([14])
            assert inverse_map.is_equivalent_to(IndexMap.from_func(lambda j,k: [4*j + k])
            print(predicate) # Prints "(axis0==3) && (axis2 >= 2)"
        """

        shape = [dim if isinstance(dim, Range) else Range(0, dim) for dim in shape]
        return _ffi_api.IndexMapNonSurjectiveInverse(self, shape)
