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
"""Source acquisition and declaration/body execution for registered builders."""

from __future__ import annotations

import ast
import dis
import inspect
import linecache
import sys
from collections import ChainMap
from collections.abc import Callable, Mapping, Sequence
from functools import wraps
from types import CodeType, FrameType, FunctionType
from typing import TYPE_CHECKING, Any, TypeVar

from tvm.ir import SourceName, Span
from tvm.script import ir_builder as builder_ir
from tvm.script.ir_builder import base
from tvm.script.ir_builder.base import SpanEntry

from . import _NAMESPACES, _initialize, jit_support
from . import protocol_registry as syntax_protocol
from . import register_namespace as register_namespace
from .inspect_source import (
    _AnnotationScope,
    acquire_source,
    capture_annotation_bindings,
    capture_definition_scope,
    capture_lexical_bindings,
    require_definition_site,
)
from .prescan import PrescanCollector
from .transpile import FunctionContext, GeneratedBuilder, IRBuilderTranspiler, ModuleContext

if TYPE_CHECKING:
    from tvm.ir import IRModule


_Callable = TypeVar("_Callable", bound=Callable[..., Any])


def _read_closure_values(function: FunctionType) -> dict[str, Any]:
    values = {}
    for name, cell in zip(function.__code__.co_freevars, function.__closure__ or ()):
        try:
            values[name] = cell.cell_contents
        except ValueError:
            # Recursive and later-bound locals are empty until the helper is used.
            pass
    return values


def _read_lexical_environment(obj: FunctionType | type) -> dict[str, Any]:
    """Retain Python globals and closure bindings without inspecting callers."""
    if inspect.isfunction(obj):
        return {**obj.__globals__, **_read_closure_values(obj)}
    module = inspect.getmodule(obj)
    return dict(vars(module)) if module is not None else {}


def _recompose_builder(
    translated: ast.Module,
    *,
    source_fn: str | FunctionType | type,
    definition_scope: Mapping[str, Any],
    filename: str,
    flags: int,
    name: str,
    make_fresh_name: Callable[[str], str],
    environment: Mapping[str, Any],
    result: str | None = None,
    generated_builders: Sequence[GeneratedBuilder] = (),
) -> Callable[..., Any]:
    """Compile one builder callable with source lexical and annotation scopes.

    Python code objects identify the source's globals and closure cells. The
    generated body keeps those lexical bindings, while annotation expressions
    execute in separate definition-site scopes inside builder declaration frames.
    """
    namespace = dict(environment)
    # Capture loaded source names and the exact requested keys of lazy snapshots.
    # An arbitrary string literal must not retain a same-named definition local.
    required = {
        item.id
        for item in ast.walk(translated)
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
    }
    for item in ast.walk(translated):
        if (
            isinstance(item, ast.Call)
            and isinstance(item.func, ast.Name)
            and namespace.get(item.func.id) is _AnnotationScope
        ):
            required.update(key.value for key in item.args[0].elts)
    definition_scope = {key: value for key, value in definition_scope.items() if key in required}
    originals = (
        {key: value for key, value in vars(source_fn).items() if inspect.isfunction(value)}
        if inspect.isclass(source_fn)
        else {source_fn.__name__: source_fn}
        if inspect.isfunction(source_fn)
        else {}
    )
    definition = None
    if result is not None:
        location = translated.body[-1]
        definition = ast.copy_location(
            ast.FunctionDef(
                name,
                ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
                [
                    *translated.body,
                    ast.copy_location(ast.Return(ast.Name(result, ast.Load())), location),
                ],
                [],
                None,
            ),
            location,
        )
        if "type_params" in ast.FunctionDef._fields:
            definition.type_params = []
        translated = ast.Module([definition], [])

    text_codes = {}
    if isinstance(source_fn, str) and definition_scope:
        # Text has no original callable. Compile the wrapper before adding defaults
        # to distinguish existing prefix/class closures from execution-global reads.
        # This executes neither source setup nor decorators/annotation expressions.
        codes = [
            compile(
                ast.fix_missing_locations(translated),
                filename,
                "exec",
                flags=flags,
                dont_inherit=True,
            )
        ]
        while codes:
            code = codes.pop()
            text_codes[code.co_name] = code
            codes.extend(item for item in code.co_consts if isinstance(item, CodeType))

    for generated in generated_builders:
        body, retained = generated.body, generated.protected_names
        original = originals.get(generated.original_func_name)
        original_code = original.__code__ if original is not None else text_codes.get(body.name)
        if original_code is None:
            continue
        parameters = {argument.arg for argument in body.args.args}
        # co_names also contains attribute spellings. Only actual global
        # instructions describe a Python global binding; an attribute may have
        # the same spelling as a captured closure cell (for example C.dtype).
        global_names = {
            instruction.argval
            for instruction in dis.get_instructions(original_code)
            if instruction.opname in ("LOAD_GLOBAL", "STORE_GLOBAL", "DELETE_GLOBAL")
        }
        # Nested Python helpers/lambdas may read globals absent from the outer
        # bytecode. A definition default must not turn those reads into closures.
        nested_codes = [item for item in original_code.co_consts if isinstance(item, CodeType)]
        original_locals = set(
            original_code.co_varnames + original_code.co_cellvars + original_code.co_freevars
        )
        while nested_codes:
            code = nested_codes.pop()
            nested_codes.extend(item for item in code.co_consts if isinstance(item, CodeType))
            global_names.update(
                instruction.argval
                for instruction in dis.get_instructions(code)
                if instruction.opname in ("LOAD_GLOBAL", "STORE_GLOBAL", "DELETE_GLOBAL")
                and instruction.argval not in original_locals
            )
        global_names -= retained | parameters
        if original is None:
            # Only new defaults can redirect the already-translated text scopes.
            global_names.intersection_update(definition_scope)
        source_closure = _read_closure_values(original) if original is not None else {}
        for captured in sorted(source_closure.keys() - retained - parameters):
            value = (
                environment.get(captured, source_closure[captured])
                if captured in source_closure and inspect.isfunction(source_fn)
                else source_closure.get(captured, environment.get(captured, base.MISSING))
            )
            # Builtin defaults remain ordinary Python lookup when no explicit
            # lexical binding exists. Missing global names are never supplied by
            # the annotation definition scope.
            if value is base.MISSING:
                import builtins

                value = getattr(builtins, captured, base.MISSING)
            # Standalone lexical inputs already have their original names in the
            # execution environment. A global declaration prevents generated
            # enclosing scopes from redirecting those body references.
            if captured in namespace and namespace[captured] is value:
                global_names.add(captured)
                continue
            # A class method may close over a different value than its class's
            # same-named member. Only that concrete conflict needs an injected
            # binding; the body itself still reads the original source name.
            alias = make_fresh_name("_lexical")
            namespace[alias] = value
            reference = ast.copy_location(ast.Name(alias, ast.Load()), body)
            body.args.kwonlyargs.append(ast.arg(captured))
            body.args.kw_defaults.append(reference)
        global_names.intersection_update(
            item.id for item in ast.walk(body) if isinstance(item, ast.Name)
        )
        if global_names:
            body.body.insert(0, ast.copy_location(ast.Global(sorted(global_names)), body))

    if definition is not None:
        # A nested `global n` also makes CPython compile a module-level n=n
        # default as LOAD_GLOBAL. Inject only that conflicting default value;
        # the parameter and every source reference retain their original name.
        declared_globals = {
            key
            for item in ast.walk(translated)
            if isinstance(item, ast.Global)
            for key in item.names
        }
        defaults = []
        for key, value in definition_scope.items():
            default_name = key
            if key in declared_globals:
                default_name = make_fresh_name("_capture")
                namespace[default_name] = value
            defaults.append(ast.Name(default_name, ast.Load()))
        definition.args.kwonlyargs = [ast.arg(key) for key in definition_scope]
        definition.args.kw_defaults = defaults
    # Defaults resolve in the actual definition context, while the callable's
    # __globals__ remains the source execution namespace. Do not inject a scope
    # dictionary into those globals or retain the outer callable there.
    definition_locals = dict(definition_scope)
    exec(
        compile(
            ast.fix_missing_locations(translated), filename, "exec", flags=flags, dont_inherit=True
        ),
        namespace,
        definition_locals,
    )
    return definition_locals.pop(name)


def _is_inside_ir_module(function: FunctionType, frame: FrameType) -> bool:
    """Recognize module decorators from the enclosing class declaration."""
    local = frame.f_locals
    if local.get("__module__") != function.__module__ or "__qualname__" not in local:
        return False
    caller = frame.f_back
    if caller is None:
        return False
    # The class code starts at its first decorator, while its caller is still
    # executing the class declaration. Read only those lines, including options
    # spread across multiple lines, rather than parsing the file for each member.
    for lineno in range(frame.f_code.co_firstlineno, caller.f_lineno + 1):
        line = linecache.getline(frame.f_code.co_filename, lineno).strip()
        if line.startswith("@") and any(
            name in line for name in ("ir_module", "py_module", "rewriter")
        ):
            return True
    return False


def make_decorator(
    builder: object,
) -> Callable[..., Any]:
    """Create a function decorator with an explicit construction namespace.

    Parameters
    ----------
    builder : object
        Namespace implementing the function construction protocol.

    Returns
    -------
    decorator : callable
        Callable used through a registered namespace at the definition site,
        such as ``@T.prim_func`` or ``@T.prim_func(**options)``. Bare callable
        aliases and later application to an existing function are unsupported.

    Raises
    ------
    ValueError
        When the returned decorator receives a non-function positional value.
    SyntaxError
        When standalone source violates a parser restriction.

    Notes
    -----
    Class members retain their Python functions until module construction.
    Standalone functions immediately transpile and execute a builder program.
    Annotations must be safe to re-evaluate: eager MissingType placeholders
    are not cached, and source annotations execute in declaration frames.
    Construction errors propagate unchanged through `parse`. Public options
    pass directly to ``builder.function_``; the language variant hook owns their defaults.
    """

    def decorator(function: FunctionType | None = None, **options: Any) -> Any:
        """Parse a Python function into a function of the selected IR language variant.

        Parameters
        ----------
        function : Callable, optional
            The function to be parsed. May be omitted to use the decorator with
            keyword options, such as ``@T.prim_func(private=True)``.
        private : bool, optional
            Whether the function should be treated as private. A private
            function has no global symbol attribute; a public function has a
            global symbol matching its name. Defaults to False.
        check_well_formed : bool, optional
            Whether to check that the constructed function is well formed.
            Defaults to True.
        persistent : bool, optional
            For ``T.prim_func``, mark the resulting function as a persistent
            kernel. Defaults to False. See
            :func:`tvm.tirx.script.ir_builder.prim_func`.
        pure : bool, optional
            For ``R.function``, declare whether the function is pure, meaning
            that it has no observable side effects. Defaults to True. See
            :func:`tvm.relax.script.ir_builder.function_`.
        **options
            Keyword options are forwarded to the selected language variant's
            :func:`~tvm.script.ir_builder.parser_protocol.function_` hook,
            except ``check_well_formed``, which controls parser validation.
            Options supported by only one language variant are not shared
            between ``T.prim_func`` and ``R.function``.

        Returns
        -------
        result : PrimFunc or relax.Function or Callable
            The parsed function, or a decorator when ``function`` is omitted.
            Class members retain their Python functions until the enclosing
            module is constructed.
        """
        if function is not None and not inspect.isfunction(function):
            raise ValueError("Construction decorators require a function or keyword options")

        def apply(function: FunctionType) -> Any:
            frame = inspect.currentframe().f_back
            try:
                if frame.f_code is decorator.__code__:
                    frame = frame.f_back
                require_definition_site(function, frame, decorator)
                deferred = _is_inside_ir_module(function, frame)
                # The module root supplies one scope for all deferred members.
                definition_scope = {} if deferred else capture_definition_scope(frame)
            finally:
                del frame
            if deferred:
                return function
            result = parse(
                function,
                definition_scope=definition_scope,
                root_function_kwargs={
                    key: value for key, value in options.items() if key != "check_well_formed"
                },
                check_well_formed=options.get("check_well_formed", True),
            )
            result.__name__ = function.__name__
            return result

        return apply(function) if function is not None else apply

    return decorator


def make_macro_decorator(
    builder: object, *, preserve_return: bool = True, late_binding: bool = False
) -> Callable[..., Callable[..., Any]]:
    """Create a decorator for helpers executed in a caller's builder frames.

    Parameters
    ----------
    builder : object
        Namespace implementing construction operations for the helper body.
    preserve_return : bool, optional
        Keep helper returns as ordinary Python control flow. Default is True.
    late_binding : bool, optional
        Refresh captured closure cells on each call. Default is False.

    Returns
    -------
    decorator : callable
        Accepts definition-site applications through a registered namespace,
        with optional keyword options. The ``hygienic`` option defaults to True
        and snapshots the definition environment;
        False captures the calling environment on each invocation. Other
        keyword options are accepted but do not affect helper construction.

    Raises
    ------
    ValueError
        When the returned decorator receives a non-function positional value.
    TypeError
        When a helper invocation cannot bind its Python signature.

    Notes
    -----
    Each invocation binds arguments and defaults, transpiles the original
    body, and returns its result. It shares the caller's active construction
    frames instead of declaring an IR function. Source acquisition,
    compilation, and builder exceptions propagate to the caller.
    """

    def decorator(function: FunctionType | None = None, **options: Any) -> Callable[..., Any]:
        """Decorate a helper that constructs IR in its caller's active frames.

        Parameters
        ----------
        function : Callable, optional
            The helper function. May be omitted to supply keyword options.
        hygienic : bool, optional
            Whether the helper resolves symbols in its definition environment
            instead of its calling environment. Defaults to True. ``T.macro``
            and ``R.macro`` capture values at definition time; ``T.inline``
            refreshes captured closure cells when called.
        **options
            Keyword configuration for this helper decorator. ``hygienic``
            controls name lookup as described above; other options are
            accepted but are not consumed or forwarded to builder hooks.

        Returns
        -------
        result : Callable
            The construction helper, or a decorator when ``function`` is omitted.

        Notes
        -----
        ``T.inline`` follows Python lexical scoping with late binding of captured
        closure cells. Its return statements produce Python values, as do those
        of ``R.macro``. ``T.macro`` emits returns in the active primitive function.

        Examples
        --------
        An inline helper can read values from its enclosing scope::

            import tvm
            from tvm.script import tirx as T
            x_value = 128

            @T.inline
            def capture(A, B):
                B[()] = A[x_value]  # x_value resolved from enclosing scope

            @T.prim_func
            def use(A: T.Buffer((1024,), "int32"), B: T.Buffer((), "int32")) -> None:
                capture(A, B)       # Produces B[()] = A[128]
        """
        if function is not None and not inspect.isfunction(function):
            raise ValueError("Construction decorators require a function or keyword options")

        def apply(function: FunctionType) -> Callable[..., Any]:
            frame = inspect.currentframe().f_back
            try:
                if frame.f_code is decorator.__code__:
                    frame = frame.f_back
                require_definition_site(function, frame, decorator)
                definition_scope = capture_definition_scope(frame)
            finally:
                del frame
            definition_env = capture_lexical_bindings(function)
            definition_scope = capture_annotation_bindings(function, definition_scope)

            @wraps(function)
            def invoke(*args: Any, **kwargs: Any) -> Any:
                bound = inspect.signature(function).bind(*args, **kwargs)
                bound.apply_defaults()
                environment = (
                    {
                        **function.__globals__,
                        **definition_env,
                        **(_read_closure_values(function) if late_binding else {}),
                    }
                    if options.get("hygienic", True)
                    else {**function.__globals__, **inspect.currentframe().f_back.f_locals}
                )
                return _run_statements(
                    function,
                    builder,
                    {**environment, **bound.arguments},
                    set(bound.arguments),
                    preserve_return=preserve_return,
                    definition_scope=definition_scope,
                )

            return invoke

        return apply(function) if function is not None else apply

    return decorator


@syntax_protocol.declaration_kind("I.pyfunc", "helper")
def pyfunc(function: _Callable) -> _Callable:
    """Keep an ordinary Python callable for collection in a module.

    Parameters
    ----------
    function : callable
        Python function to retain in the module.

    Returns
    -------
    callable
        The same function, unchanged.

    Notes
    -----
    The function body remains ordinary Python. Shared module parsing attaches
    it to the result's ``__pyfuncs__`` mapping. This decorator enters no frame and
    preserves callable identity for the function's lifetime.
    """
    return function


def _prepare_transpiler(
    tree: ast.Module,
    source: str | FunctionType | type,
    environment: Mapping[str, Any],
    definition_scope: Mapping[str, Any],
    filename: str,
    *,
    track_span: bool = True,
    enable_jit_map: bool = False,
    root_function_kwargs: Mapping[str, Any] | None = None,
    **options: Any,
) -> tuple[IRBuilderTranspiler, dict[str, Any]]:
    """Prescan an owned tree and inject collision-free execution bindings.

    The lexical environment is copied per invocation. Descriptor-safe metadata
    lookup retains namespace owners, including annotation-only definition
    bindings; ordinary body values remain opaque. Prescan facts are read-only, while the
    local name map allocates fresh identifiers across the complete source unit.
    No builder frame or expression is created here.
    """
    namespace = {
        "tvm": sys.modules.get("tvm"),
        **_NAMESPACES,
        **environment,
    }
    # Imports establish source-text namespace metadata before prescan. Their
    # original AST nodes remain owned here and execute only once.
    imports: list[ast.stmt] = [
        node for node in tree.body[:-1] if isinstance(node, ast.Import | ast.ImportFrom)
    ]
    if imports:
        exec(compile(ast.Module(imports, []), filename, "exec", dont_inherit=True), namespace)
    # Fixed namespace lookup references the one root definition scope directly.
    # A class namespace has precedence, matching its source definition context.
    metadata = ChainMap(
        vars(source) if inspect.isclass(source) else {}, definition_scope, namespace
    )
    # Construction policy comes from the source declaration's namespace.
    prescan_ctx = PrescanCollector(metadata, filename=filename).collect(tree)
    names = dict.fromkeys([*namespace, *prescan_ctx.reserved_names], 0)

    def make_fresh_name(prefix: str = "_t") -> str:
        """Allocate a name without changing any source identifier."""
        counter = names.get(prefix, 0)
        while f"{prefix}{counter}" in names:
            counter += 1
        name = f"{prefix}{counter}"
        names[prefix], names[name] = counter + 1, 0
        return name

    builder_name, ir_prefix = make_fresh_name("_X"), make_fresh_name("_I")
    namespace[ir_prefix] = builder_ir
    span_table_name = make_fresh_name("_S") if track_span else None
    if track_span:
        # Entries contain fixed native metadata only. The existing rewrite creates
        # them on demand; there is no location collection pass or retained AST.
        source_name = SourceName(filename)
        span_entries: list[SpanEntry] = []
        span_indices: dict[tuple[int, int, int, int], int] = {}
        namespace[span_table_name] = span_entries

    def make_span_expr(node: ast.AST) -> ast.expr:
        """Materialize a needed location and emit its injected table reference."""
        if not track_span:
            return ast.copy_location(ast.Constant(None), node)
        coordinates = (
            node.lineno,
            node.end_lineno,
            node.col_offset + 1,
            node.end_col_offset + 1,
        )
        index = span_indices.get(coordinates)
        if index is None:
            index = len(span_entries)
            span_indices[coordinates] = index
            span_entries.append(SpanEntry(Span(source_name, *coordinates)))
        location = ast.Subscript(
            ast.Name(span_table_name, ast.Load()), ast.Constant(index), ast.Load()
        )
        return ast.copy_location(location, node)

    context = ModuleContext(
        filename,
        metadata,
        ir_prefix,
        make_span_expr,
        make_fresh_name,
        track_span=track_span,
        enable_jit_map=enable_jit_map,
        definition_scope=definition_scope,
        original_func_map=(
            {key: value for key, value in vars(source).items() if inspect.isfunction(value)}
            if inspect.isclass(source)
            else {source.__name__: source}
            if inspect.isfunction(source)
            else {}
        ),
        prescan_ctx=prescan_ctx,
        exec_globals=namespace,
        root_function_kwargs=root_function_kwargs,
    )
    transformer = IRBuilderTranspiler(
        context, FunctionContext(options.pop("current_scope", None), builder_name), **options
    )
    return transformer, namespace


def _run_statements(
    source: FunctionType,
    builder: object,
    environment: Mapping[str, Any],
    bound_names: set[str],
    *,
    preserve_return: bool = False,
    definition_scope: Mapping[str, Any] | None = None,
) -> Any:
    """Execute a macro body in its caller's active builder frames.

    Argument binding precedes this call. The helper owns the freshly acquired source AST,
    keeps Python parameter names and optionally keeps ordinary Python returns.
    Compilation uses the original coordinates without unparse/reparse. Builder
    and host exceptions propagate unchanged to the caller.
    """
    tree, filename, flags = acquire_source(source)
    definition_scope = {} if definition_scope is None else definition_scope
    transformer, namespace = _prepare_transpiler(
        tree,
        source,
        environment,
        definition_scope,
        filename,
        preserve_return=preserve_return,
        current_scope=tree.body[-1],
    )
    namespace[transformer.function.dialect_prefix] = builder
    node = tree.body[-1]
    statements = transformer.transform_statements(node.body)
    names = sorted(name for name in bound_names if name in namespace)
    helper_name = transformer.module.make_fresh_name("_macro")
    helper = ast.copy_location(
        ast.FunctionDef(
            helper_name,
            ast.arguments(
                posonlyargs=[],
                args=[ast.arg(name) for name in names],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            statements or [ast.Pass()],
            [],
            None,
        ),
        node,
    )
    if "type_params" in ast.FunctionDef._fields:
        helper.type_params = []
    runnable = _recompose_builder(
        ast.Module([helper], []),
        source_fn=source,
        definition_scope=transformer.module.definition_scope,
        generated_builders=transformer.module.generated_builders,
        filename=filename,
        flags=flags,
        name=helper_name,
        make_fresh_name=transformer.module.make_fresh_name,
        environment=namespace,
    )
    return runnable(*(namespace[name] for name in names))


def parse(
    source: str | FunctionType | type,
    extra_vars: Mapping[str, Any] | None = None,
    *,
    filename: str | None = None,
    track_span: bool = True,
    definition_scope: Mapping[str, Any] | None = None,
    root_function_kwargs: Mapping[str, Any] | None = None,
    **options: Any,
) -> Any:
    """Transpile and execute a source string, Python function, or Python class.

    Parameters
    ----------
    source : str or function or type
        Original source text or inspectable Python object.
    extra_vars : mapping of str to object, optional
        Lexical bindings overriding captured values. Default is None,
        interpreted as an empty mapping.
    filename : str, optional
        Source filename override. Default is None, which uses the inspected
        filename for objects and ``"<str>"`` for text.
    track_span : bool, optional
        Enable shared source metadata and IR location instrumentation.
        Default is True. False retains Python source locations only.
    definition_scope : mapping of str to object, optional
        Temporary definition-site bindings for annotation reconstruction. None
        and an empty mapping both add no external scope; parse never inspects its
        caller for bindings. These values do not replace body globals or closures.
    root_function_kwargs : mapping of str to object, optional
        Already-evaluated kwargs forwarded to ``function_`` for a standalone
        root function. None reads options from its source decorator; an empty
        mapping supplies no kwargs and does not re-evaluate source arguments.
        Nested functions and module members use their own source decorators.
        Defaults are owned by the builder hook; check_well_formed is separate.
    **options
        ``_specialization_bindings`` is a mapping of selected constexpr values
        and explicit optional-parameter absence. None selects ordinary parsing;
        an empty mapping still selects root JIT construction. ``check_well_formed``
        controls completed-result validation; construction policy otherwise
        comes from qualified source decorators.

    Returns
    -------
    object
        Opaque result of the generated builder program.

    Raises
    ------
    OSError
        If source inspection cannot recover the supplied object's text.
    TypeError
        If the source object cannot be inspected.
    SyntaxError
        If source parsing fails or transpilation detects a syntax restriction.
        Parser restrictions retain their source filename and range.

    Notes
    -----
    Each call owns its freshly acquired AST and a fresh lexical environment. Declaration and
    definition frames are entered only during generated execution. Source
    acquisition, host and builder errors propagate with their original type,
    identity and traceback. Temporary captures are released even when execution
    fails.
    """
    # Direct entry.parse callers need the same registered namespaces as public entry.
    _initialize()
    # - Recover source and explicit lexical/definition inputs.
    # - Collect source syntax facts on this invocation's freshly acquired AST.
    # - Rewrite syntax into a builder program and recompose its lexical bindings.
    # - Execute the private builder immediately and release temporary captures.
    # Definition scope is a per-root input; it never replaces body globals/closures.
    env = {} if isinstance(source, str) else _read_lexical_environment(source)
    env.update(extra_vars or {})
    definition_scope = {} if definition_scope is None else definition_scope
    tree, filename, flags = acquire_source(
        source, filename, definition_source=options.pop("_definition_source", None)
    )
    # Acquisition returns a fresh tree; prescan and rewriting own it directly.
    _builder = None
    try:
        root = tree.body[-1]
        root_name = root.name if isinstance(root, ast.FunctionDef) else None
        specialization = options.get("_specialization_bindings")
        check_well_formed = options.get("check_well_formed")
        if check_well_formed is None:
            check_well_formed = True
            for decorator in getattr(root, "decorator_list", ()):
                if isinstance(decorator, ast.Call):
                    for keyword in decorator.keywords:
                        if keyword.arg == "check_well_formed":
                            check_well_formed = eval(
                                compile(ast.Expression(keyword.value), filename, "eval"),
                                {**_NAMESPACES, **env},
                            )
        # Prescan and rewrite consume only the owned syntax and fixed metadata.
        transformer, namespace = _prepare_transpiler(
            tree,
            source,
            env,
            definition_scope,
            filename,
            track_span=track_span,
            enable_jit_map=specialization is not None and root_name is not None,
            root_function_kwargs=root_function_kwargs,
        )
        transformed, result_name = transformer.rewrite_module(
            tree, check_well_formed=check_well_formed
        )
        # Recomposition preserves original source ranges and body globals/closures.
        _builder = _recompose_builder(
            transformed,
            source_fn=source,
            definition_scope=transformer.module.definition_scope,
            generated_builders=transformer.module.generated_builders,
            filename=filename,
            flags=flags,
            name=transformer.module.make_fresh_name("_builder"),
            make_fresh_name=transformer.module.make_fresh_name,
            environment=namespace,
            result=result_name,
        )
        # Important: do not retain _builder. Its globals and closures may keep
        # values from the enclosing scope alive.
        with jit_support.use_specialization(root_name, specialization):
            result = _builder()
        return result
    finally:
        # Release temporary captures on both successful and exceptional exits.
        _builder = None
        definition_scope = None


def ir_module(module: type | None = None, **options: Any) -> IRModule | Callable[[type], IRModule]:
    """Decorate a Python class with two-phase module construction.

    Parameters
    ----------
    module : type, optional
        Class to compile immediately. Default is None, which returns a
        decorator awaiting a class.
    **options
        Keyword arguments forwarded to `parse`.

    Returns
    -------
    object or callable
        Generated module result when a class is supplied, otherwise a class
        decorator.

    Raises
    ------
    SyntaxError
        If module source violates a parser restriction.

    Notes
    -----
    Class host bindings are captured before transpilation. Generated execution
    declares all registered signatures before defining their bodies. Source
    acquisition and builder errors propagate unchanged; frame lifetime follows
    `parse`.
    """

    def apply(module: type) -> IRModule:
        if not inspect.isclass(module):
            raise TypeError(f"Expect a class, but got: {module}")
        frame = inspect.currentframe().f_back
        try:
            if frame.f_code is ir_module.__code__:
                frame = frame.f_back
            definition_scope = capture_definition_scope(frame)
            definition_source = (frame.f_code.co_filename, frame.f_lineno)
        finally:
            del frame
        result = parse(
            module,
            definition_scope=definition_scope,
            _definition_source=definition_source,
            **options,
        )

        result.__name__ = module.__name__
        return result

    return apply(module) if module is not None else apply


syntax_protocol.module_decorator("I.ir_module")(ir_module)
syntax_protocol.module_decorator("script.ir_module")(ir_module)

from_source = parse
