# Informal Relax Language Specification

Note: Text in «double chevrons» indicates features not present in the current prototype.

In order to develop and test Relax, it is important for compiler developers to agree on what a given program in Relax means and what makes it valid so that test cases can be evaluated independently of any particular Relax implementation. This document is intended to describe Relax's grammar constructs (its [abstract syntax tree](https://en.wikipedia.org/wiki/Abstract_syntax_tree), or AST), the semantics of its grammar (what the different constructs mean), Relax's type system and type-checking rules (what makes a Relax program valid), and its rules for reasoning about structural information (such as tensor shapes) in detailed though still informal terms. If necessary, we may encode these rules more formally to allow for more automated analysis.

Though this document will use the TVMScript front end for some examples, specifying the mapping from Python's AST to Relax's AST will be deferred until the parser becomes more stable.

# Table of Contents

1. [Overview](#overview)
2. [Top-Level Program Organization](#top-level-program-organization-irmodule)
3. [Values in Relax](#values-in-relax)
4. [Variable Scoping](#variable-scoping)
5. [Normal Form](#normal-form)
6. [Well-Formedness Criteria](#well-formedness-criteria)
7. [Structural Information in Relax](#structural-information-in-relax)
8. [Semantics](#detailed-semantics)

# Overview

This section will outline the grammar of Relax and give very brief descriptions of the different components, including the semantics and structural information (`StructInfo`) system. The rest of this document will provide more detailed descriptions of these facets of the language, including the validity conditions that the `StructInfo` system upholds.

## Differences from Relay

Per the [original workshop paper](https://arxiv.org/abs/1810.00952) and the [later report](https://arxiv.org/abs/1904.08368), Relay was designed to be a high-level functional language for expressing deep learning models at a high level. While Relay is not entirely pure (the `Ref` type is modeled after reference types in SML and similar functional languages), the assumption in Relay is that tensor operators are generally pure, meaning that they do not change the program state other than by producing new values. Additionally, Relay's type system also requires operators to have type relations that infer static tensor types or conclude that a dimension is unknown at compile time (`Any`). The need to register type relations and ensure operators' purity makes it difficult to add new operators to Relay and particularly difficult to call directly into TIR or external libraries, which are often not pure; any such extension requires adding new operators and abstracting over any impurity.

While Relax aims to be as general and expressive as Relay, Relax is intended to make it much easier to interoperate with external libraries and especially with TIR. In particular, Relax includes a mechanism for calling arbitrary TVM `PackedFunc`s (which can call external libraries) and special support for TIR. The language accordingly does not assume that such operations are pure, though this does require reasoning about aliasing and similar issues. Additionally, tensor shapes are no longer handled during type checking; each expression has associated structural information associated with it, in addition to a type. This structural information supports static reasoning about tensor shapes in many cases, but also facilitates a fallback to dynamic checking when that is not possible. This approach to shapes allows for richer shape constraints and other structural properties to be checked at run time (such as with _symbolic_ shapes, where some dimensions are variables) and allows for more quickly integrating calls into TIR or external libraries into Relax code by obviating the need for type relations.

## Grammar

Below is a diagram of the various AST constructs in Relax, including types. In code, these are defined on the C++ side in `include/tvm/relax/{expr.h, type.h}` and in Python in `python/tvm/relax/{expr.py, ty.py}`. This diagram will give the names of the AST nodes and the types and names of their members. The semantics will describe what computation each construct represents; an AST is simply data. A Relax program consists of an `IRModule` with global variables bound to Relax functions that implement the computations of interest.

(On the notation: `[x]` means "a list of `x`," `x?` means "optionally `x`," `{x: y}` means "a map of `x` to `y`," `x | y` means "`x` or `y`," and `#` is used for comments. For the definition of `PrimFunc`, AST constructs are prefixed with `tir::` to indicate that these are the TIR versions of these AST nodes rather than the Relax ones.)

```
# PrimExprs are defined in TIR, see include/tvm/tir/expr.h
# They are intended to have the same semantics as in TIR
PrimExpr ::=
           Var(name: string) # shape variables
         | IntImm(value: int64)
         | Add(a: PrimExpr, b: PrimExpr)
         | Sub(a: PrimExpr, b: PrimExpr)
         | Mul(a: PrimExpr, b: PrimExpr)
         | Div(a: PrimExpr, b: PrimExpr)
         | Min(a: PrimExpr, b: PrimExpr)
         | Max(a: PrimExpr, b: PrimExpr)
         | Not(a: PrimExpr)
         | And(a: PrimExpr, b: PrimExpr)
         | Or(a: PrimExpr, b: PrimExpr)
         | Select(condition: PrimExpr, true_value: PrimExpr, false_value: PrimExpr)
         # (others may be added later, as deemed necessary)

# See include/tvm/tir/function.h
# Can appear at the module level but otherwise do not interact with any Relax constructs;
# intended to have the same semantics as in TIR
PrimFunc ::= PrimFunc(params: [tir::Var], body: tir::Stmt, ret_type: tir::Type?,
                      buffer_map: {tir::Var: tir::Buffer}, attrs: Attrs)


# VDevice is used to indicate target devices for heterogeneous computing
Target ::= Target() # null target
         | Target(tag: string)
         | Target(config: {String, ObjectRef})

VDevice ::= VDevice(tgt: Target, int: dev_id, mem_scope: string)

# Also from TIR
DataType ::= Int(bits: int, lanes: int)
           | UInt(bits: int, lanes: int)
           | Float(bits: int, lanes: int)
           | Handle(bits: int, lanes: int)


StructInfo ::= TensorStructInfo(shape: Expr?, dtype: DataType, vdevice: VDevice?, ndim: int)
             | ShapeStructInfo(values: [PrimExpr]?, ndim: int)
             | PrimStructInfo(dtype: DataType, value: PrimExpr?)
             | ObjectStructInfo()
             | TupleStructInfo(fields: [StructInfo])
             | FuncStructInfo(params: [StructInfo]?, ret: StructInfo, purity: bool, derive_func: EnvFunc?*)

# expressions
Expr ::=   Constant(data: NDArray)
           # scoped to functions or SeqExprs
         | Var(name_hint: string, struct_info_annotation: StructInfo?)
           # scoped to DataflowBlocks
         | DataflowVar(name_hint: string, struct_info_annotation: StructInfo?)
         | GlobalVar(name_hint: string)
         | Tuple(fields: [Expr])
         | SeqExpr(blocks: [BindingBlock], body: Expr)
         | PrimValue(value: PrimExpr)
         | StringImm(value: string)
         | DataTypeImm(value: DataType)
         | Function(params: [Var], body: Expr, ret_struct_info: StructInfo?, is_pure: bool?, attrs: Attrs?)
         | If(cond: Expr, true_branch: Expr, false_branch: Expr)
         | ExternFunc(global_symbol: string)
         | Call(op: Expr, args: [Expr], sinfo_args: [StructInfo], attrs: Attrs?)
         | ShapeExpr(values: [PrimExpr])
         | TupleGetItem(tuple_value: Expr, index: int)
         | Op(op_name: string)

# binding blocks (analogous to sequence of statements)
BindingBlock ::= 
           BindingBlock(bindings: [Binding])
         | DataflowBlock(bindings: [Binding])

# bindings (analogous to statements)
Binding ::= 
           VarBinding(var: Var|DataflowVar, value: Expr)
         | MatchCast(var: (Var|DataflowVar)?, struct_info: StructInfo, value: Expr)

# Relax programs are IRModules. Modules may bind global variables either to
# Relax functions or TIR PrimFuncs.
# The Relax compiler may analyze and modify the TIR PrimFuncs as well.
# Note that there can be global info other than VDevices, but only VDevice
# is used by Relax at present.
Program ::= IRModule(
              funcs: {GlobalVar: Function|PrimFunc}
              global_info: {string: VDevice}
            )
```

### Notes on `derive_func`

The `derive_func` field of `FuncStructInfo` is a macro in the meta-language: Given a function call and the variable mapping context, return the `StructInfo` of the result. This field is used only at compile time for reasoning about the `StructInfo` of calls to `ExternFunc`s.

### Notes on `DataType` and Related Terminology

The representation of datatypes, `DataType`, in the above AST is taken directly from TIR. However, the usage of datatypes in Relax is more restricted than in TIR.
1. The `lanes` field for the `Int`, `UInt`, and `Float` datatypes must always be 1; we do not directly consider vectorized values in Relax.
2. The `bits` field for the `Handle` datatype must always be 0, indicating that it is `Void` (see below). The `lanes` field for `Handle` should always be set to 0 (it will not be used by Relax).

We also define the following special notation for datatypes, to be used in the rest of the specification:
1. `Bool()`: This is shorthand for `UInt(bits=1, lanes=1)`, since TIR does not have a separate Boolean type. "True" refers to a value of 1 in this datatype and "false" refers to a value of 0. For convenience, we will refer to Boolean values as a separate datatype in the specification, due to their significance in `If` nodes.
2. `Void()`: This is shorthand for `Handle(bits=0, lanes=0)`. TIR uses this datatype to refer to opaque objects; in Relax, it is used to denote an unknown datatype.

## Expression Survey

This specification provides a more detailed description of what each expression and `StructInfo` represents and what conditions make them valid. To motivate and provide more context for the full specification later in this document, this section will briefly summarize the purpose of each node.

1. `Constant` nodes construct tensor constants (n-dimensional arrays of scalars).
2. `Tuple` nodes construct a tuple (immutable fixed-size ordered grouping) of Relax values.
3. `Var`, `DataflowVar`, and `GlobalVar` nodes are all variables, referring to named stored values of different kinds. Variables in Relax must be bound exactly once. `GlobalVar`s are bound in the `IRModule` itself and refer to Relax functions or TIR `PrimFunc`s. `Var` nodes are bound either within functions, where they represent function parameters, or in `VarBinding` or `MatchCast` nodes in `BindingBlock`s, as we will discuss below. `DataflowVar`s are similar to `Var`s and can be bound only within `DataflowBlock`s.
4. `PrimExpr`s are used to represent dimensions of shapes in `ShapeExpr` and `MatchCast` nodes. These represent operations on integers with their own `Var` nodes (`tir::Var`), which we will refer to as "shape variables". Shape variables can only be used in other `PrimExpr`s and are scoped like `Var` nodes (`relax::Var`), which we will call "Relax variables."
5. `ExternFunc` nodes evaluate into `PackedFunc`s; the implementation will look up the registered `PackedFunc` by its global symbol.
6. `PrimValue` nodes construct immutable scalar values from `PrimExpr`s, primarily for interacting with `ExternFunc`s or operators. These scalars are boxed within TVM objects, allowing them to be nested inside TVM's containers. (By contrast, zero-dimensional tensors defined via `Constant` are mutable.)
7. `StringImm` nodes construct strings, intended primarily for interacting with `ExternFunc`s or operators.
8. `DataTypeImm` nodes construct representations of TIR datatypes, intended primarily for interacting with `ExternFunc`s or operators (e.g., for TIR intrinsics that take a datatype as an input).
9. `Call` nodes represent function calls. The callee argument (the `op`) can be an `ExternFunc` node (representing a call to a `PackedFunc`), an `Op` node (representing a call to a Relax operator), or an arbitrary expression. 
    1. `Op` nodes refer to built-in Relax operators, which the compiler is free to implement as is deemed appropriate. Certain operators implement important operations, like `call_tir` (allows for calling TIR `PrimFunc`s).
    2. Any other expression must evaluate to a `PackedFunc` or a closure; the result of evaluating `op` will then be called with the given arguments. 
    
    Calls to `ExternFunc`s and operators may perform side effects, hence it is important to reason about whether a function call is permitted inside a `DataflowBlock`.
    
10. `If` nodes represent branching control flow. First the condition expression is evaluated, and it must evaluate to a Boolean scalar. If the condition is true, the true branch is evaluated and its result is used; otherwise, the false branch is evaluated and its result is used.
11. `TupleGetItem` nodes represent tuple indexing. The `tuple_value` expression must evaluate to a tuple with at least `index + 1` items and the item with the given index will be returned.
12. `SeqExpr` describes a sequence of binding blocks followed by a return expression. The `SeqExpr` opens a new scope. Its binding blocks are evaluated in order and add new variables to the scope. Binding blocks are either ordinary `BindingBlock`s or `DataflowBlock`s and both consist of a series of bindings. `DataflowBlock`s are the only kind allowed to introduce bindings with `DataflowVar`s and it does not permit any constructs featuring control flow (`If` nodes or recursive calls) or calls to (possibly) impure functions. There are two different kinds of bindings:
    1. `VarBinding`s: The `value` expression (the right-hand side of the binding) of the binding is evaluated first and is bound to the `var` expression, which must be a new `Var` or `DataflowVar` (in a dataflow block). The newly bound variable will have that value for the remainder of the scope: `DataflowVar`s are scoped only to any later bindings in the `DataflowBlock` in which they were defined; `Var`s are scoped to any later bindings within the `BindingBlock` in which they were defined, as well as any bindings in subsequent `BindingBlock`s in the `SeqExpr` and in the `body` field of the `SeqExpr`.
    2. `MatchCast`s: The `value` expression is evaluated and the result is dynamically checked against the structural information given in the `struct_info` field.
        1. The types must match: All `StructInfo` variants correspond to a category of value value (`TensorStructInfo` to a tensor value, `ShapeStructInfo` to shape values, etc.), so if the structure of `value` does not correspond to `struct_info`, an error is triggered. The structure of `value` is compared recursively with `struct_info`, so all components of `value` must match up with any nested structural information. Special comparison rules:
            1. For comparing tensor values to `TensorStructInfo`, `ndim` must match the number of dimensions in the tensor value (unless `ndim` is -1) and `dtype` must match the datatype used (unless `dtype` is `Void`). If `shape` has been specified, the shape of the value must match that encoded by `shape`; if specified, `shape` must be either a `Var` already bound in the current scope or a `ShapeExpr`.
            2. For comparing shape values to `ShapeStructInfo`, `ndim` must match the number of dimensions in the shape value (unless `ndim` is -1). If `values` has been specified, the shape value must match that encoded by `values`.
            3. «For comparing closures (function values) to `FuncStructInfo`, it is necessary for the compiled program to track run-time structural information for closures, since it is not possible to introspect the closure; this subject will be discussed in further detail later in the document.»
        2. When comparing tensor values with `TensorStructInfo` or shape values with `ShapeStructInfo`, any member of `shape` in `TensorStructInfo` (if `shape` is a `ShapeExpr`) or `values` in `ShapeStructInfo` that consists of a single new (hitherto unbound) shape variable is treated as a binding: The shape variable is bound to the size of the corresponding dimension of the value being matched. 
        3. If there is a variable provided, the value is bound to the `var` expression (if the variable is omitted, the structural check is performed and any shape variables are updated, but no new binding is introduced). Shape variables introduced in a `SeqExpr` are similarly scoped to the `SeqExpr`.
    
    The `SeqExpr`'s `body` expression is allowed to reference any `Var`s introduced within the `SeqExpr`'s binding blocks in addition to those that were in the outer scope; the `body` expression is evaluated after the binding blocks and its value is what is returned. Any Relax variables and shape variables introduced in the `SeqExpr` are removed from scope after the expression finishes evaluating.
    
13. `ShapeExpr` nodes construct shape literals, which are immutable collections of shape dimensions. The `PrimExpr`s within it describe how to compute each dimension; they are free to use any shape variables that are in scope.
14. `Function` nodes represent function definitions, taking in the listed parameters and evaluating the body expression in a new scope (meaning any variables defined from within the function cannot be referenced outside it). Function definitions may be nested in any other expression and they evaluate into closure values, ensuring that functions are first-class. Closures capture any variables from the outer scope that are used in their body, both Relax variables and shape variables. Note that function definitions themselves are anonymous—a function must be registered in the `IRModule` (bound to a `GlobalVar`) or appear on the right-hand side of a binding to have a name in order to be called recursively. 
    
    The function can have structural annotations on the parameters and a structural annotation for the return value. When the function is called, the annotations on parameters are checked against the argument values in similar fashion to `MatchCast` and can introduce new shape variables that are scoped to the function. Additionally, the structural information of the return value is checked against the annotation before the call returns.

    In addition to the structural annotations for the parameters and the return value, the `is_pure` field on a `Function` node serves to annotate whether the `Function` itself is pure (has no visible side effects) or not. The `StructInfo` system tracks purity in order to judge what calls are permitted inside `DataflowBlock`s. At this time, Relax makes no attempt to infer the purity of functions, so it is required for users to annotate the purity (if no annotation is provided, `is_pure` will be treated as true; since this is by far the most common case for deep learning applications, it is in practice necessarily to annotate purity if the function is _impure_).
    
    «A function mapped bound to a `GlobalVar` can have a `global_symbol` attribute defined to indicate that it should be externally linked externally (be accessible outside the `IRModule`). The absence of a `global_symbol` attribute on a function definition bound to a `GlobalVar` indicates that it is "private" and hence can be called only within the `IRModule`.»
    
## Purity and Dataflow Blocks

A function or operator is called "pure" if it does not have side effects, which refers to any change in program state besides returning a result, and depends only on the values of its arguments. Side effects include mutating values other than those they create, aborting the program, or file I/O (including writing to the console). Purity is a useful property for compiler optimizations, since calls to pure functions can be reordered or duplicated or (if the result is unused) eliminated without changing any other program behavior. Note that referring to external state separate from the function arguments (e.g., like the system clock) also renders a function impure; for example, the compiler would not be able to assume that it is safe to reorder the function calls (doing so could affect the results). Most deep learning operators are pure, as they perform arithmetic on tensors and return a new tensor containing the result.

Above, it is mentioned that `DataflowBlock`s are not allowed to contain constructs featuring control flow (`If` nodes or recursive calls to the current function) or calls to impure functions. This ensures that `DataflowBlock`s represent a directed acyclic graph of pure operations, which is similar to the graph-like abstractions of traditional deep learning frameworks. This allows many common optimizations from past frameworks to be directly adapted to `DataflowBlock`s without having to accommodate additional reasoning about more expressive features like control flow and side effects.

There is one visible side effect that Relax permits inside otherwise "pure" functions, namely exiting the program with an error. This can arise in the following cases:

- Casting errors (from `MatchCast` or from implicit structural information checks upon calling a Relax function)
- Errors raised by otherwise pure Relax operators or `PackedFunc`s. Since the purity of operators or `PackedFunc`s must be manually registered, this means that it is permissible to register an operator or `PackedFunc` as being pure if its only side effect is issuing an error in some cases.

Even though an abnormal program exit is a visible side effect and removing or reordering it changes the observable semantics, it would be too great a restriction to prohibit error checking inside `DataflowBlock`s. Relax does not have any notion of exception handling, so the only consequence of a failed safety check can be exiting the program. It is permissible for the compiler to reorder, duplicate, or eliminate `MatchCast`, or otherwise pure operations that have the potential of failing, provided that doing so does not change the value returned by the program or any other visible behavior.

Note that in some programming languages like Koka, non-termination is also considered a side effect, since it can in some sense be "observed" by a user and affects the visible behavior of a program (e.g., if there is an infinite loop before a print statement, the print will never happen). However, since non-termination cannot be automatically detected in general and is unlikely to arise in deep learning models, we do not attempt to systematically track non-termination in Relax. In general, the Relax compiler is allowed to reorder or remove otherwise pure function calls even if they may not terminate. For example, if a pure function `f` that returns an integer scalar does not terminate, it is permissible in principle to rewrite `f() - f()` to 0.

Exiting with an error and infinitely looping are traditionally considered "[divergence](https://en.wikipedia.org/wiki/Divergence_(computer_science))" in the programming languages literature. As a general principle, Relax's compiler is permitted to turn a program that diverges into a program that does not diverge (provided that no other visible effects change) so long as it never transforms a program that does not diverge into one that diverges.

## Structural Information (`StructInfo`) System Survey

Analogously to a type system in most languages, Relax tracks structural information (referred to as `StructInfo` in the implementation) related to the categories of values in Relax:
1. `TensorStructInfo` corresponds to tensor values, giving the scalar data type, the number of dimensions (rank), and an expression that computes the tensor's shape (either a `ShapeExpr` or a `Var`), all of which are optional. The optional `vdevice` ("virtual device") field, if present, indicates which device a tensor is located on. Tensor operators must be implemented on the appropriate device.
2. `TupleStructInfo` corresponds to tuple values, giving the `StructInfo` for each member of the tuple.
3. `PrimStructInfo` corresponds to `PrimValue`s (immutable scalar values), giving their TIR datatype.
4. `ShapeStructInfo` corresponds to shape values, optionally giving the number of dimensions in the shape and an expression that computes the shape's dimensions (either a `ShapeExpr` or a `Var`).
5. `FunctionStructInfo` corresponds to function values (closures) and `PackedFunc`s (external functions), giving the types of the parameters, the return type, and whether the function is pure.
6. `ObjectStructInfo` is a parent to all Relax `StructInfo` and corresponds to all the values above as well as any values returned by `PackedFunc` calls that do not fit in the above categories.

`StructInfo` is assigned to every variable in scope and every type of expression based on the values it returns via a set of inference rules defined later in the specification, making use of subtyping to assign more general `StructInfo` when a more specific one cannot be determined. «Relax is strongly typed, meaning that if the `StructInfo` inferred is less specific than the one expected, an error will be issued and an explicit check via `MatchCast` will be required.»

In Relax, tensor shapes are not statically handled in the type system, even though it would be greatly beneficial for the compiler to make use of shape information for static optimizations. Instead, shape information is tracked using Relax's structural information system, in which every expression has structural information associated with it (like tensor shapes) that is more expressive than its type. `StructInfo` can convey richer properties about expressions, like tensor shapes, and can facilitate a greater degree of static reasoning. However, when it is not feasible for the compiler to draw conclusions about structural information, this information can be checked dynamically via `MatchCast`. The structural information is essentially an extended type system, so `MatchCast` also serves to handle type casting.

---

# Top-level Program Organization: `IRModule`

As with Relay, the top level of organization for a Relax program is an `IRModule`. An `IRModule` contains mappings of global variables to functions, both Relax functions as well as TIR functions (which can be called from Relax). The global function called `main` is usually considered the entry point to the program (meaning that execution starts by calling that function), though any function with a `global_symbol` attribute can be specified as the entry point during compilation. In the AST (see below), the names of Relax functions in the `IRModule`s are `GlobalVar` nodes.

Oftentimes, compiler passes operate only on particular functions or add new functions to the `IRModule`, but a pass can operate over the entirety of a Relax program by iterating through all the functions in an `IRModule`.

# Values in Relax

Here are the classes of values that Relax operates over, meaning that they can be assigned to variables or be the result of evaluating expressions.

- *Tensors* are n-dimensional arrays of scalar values (which can be signed or unsigned integers of fixed bitwidths, floats of fixed bitwidths, or Boolean values). A tensor's *shape* is a tuple of the size of each dimension; the number of dimensions is a tensor's *rank*. For example, a vector (1, 2, 3) is a rank-1 tensor of shape `(3,)`. Note that scalars are tensor values with a rank of 0, meaning that their shape is `()`. Tensors can be _located_ on different devices in the program, namely one of the `VDevice`s listed in the `IRModule`'s `global_info` map; if tensors are located on different devices, it may be necessary to insert operators like `to_vdevice` in order to transfer them so that they can be used together in an operator call.
- *Tuples* represent a fixed-size immutable grouping of other Relax values (tensors, closures, shapes, objects, or other tuples, to an arbitrary degree of nesting). Note that an empty tuple, i.e., `()`, also called "unit" in functional programming, is commonly used as the return value for operations not intended to return a value (as may be the case in some `PackedFunc` or operator calls that have side effects).
- *Closures* are the values resulting from evaluating Relax function expressions; closures can be passed around like other values, ensuring that functions are first-class in Relax. Functions defined in Relax can capture variables from outer scopes. A [closure](https://en.wikipedia.org/wiki/Closure_(computer_programming)) consists of a function and a mapping of any variables "captured" (those are *free variables* in the function body, variables from an outer scope that are neither arguments nor defined within the function but are used in the function) to their values. Closures capture both Relax-level local variables and shape variables from outer scopes. A closure also stores a name for itself when the body contains recursive calls. «Closures additionally carry some *run-time structural information* (RTSI) indicating their argument and result structures, in order to facilitate dynamic structural checks (since it is not otherwise possible to introspect the function contained within a closure); the precise form of the RTSI is left up to the compiler implementation to determine so long as `MatchCast` can verify the structure of a closure, including whether it is pure. Closures can be evaluated in a call node, which results in calling the function with the call's arguments and the captured values.»
- *Tensor shapes* (shape values) are immutable tuples of integers describing a tensor shape, obtained by evaluating `ShapeExpr`s.
- *Packed functions* (`PackedFunc`s or external functions) represent arbitrary opaque functions implemented in TVM. That is, packed functions are routines that are defined outside of Relax and cannot be inspected by the compiler. They can perform side effects and return arbitrary values.
- *TIR `PrimFuncs`* are functions in TIR. They are usually invoked using the `call_tir` operator, but can be called on their own as first-class functions.
- *Primitive values* (`PrimValue`s) represent immutable scalar values that are primarily intended for being passed to external procedures, like calls to `PackedFunc`s. As a rule of thumb, scalar values intended for arithmetical computations should be 0-rank tensors while scalar values meant to serve as metadata should be `PrimValue`s.
- Additionally, there are further  *arbitrary objects* that do not belong in the above categories. These can be returned by `PackedFunc`s and operators. Though Relax expressions other than `PackedFunc` and operator calls cannot use those objects, Relax should pass around these values faithfully. In the future we may add more value types in order to distinguish between different objects, but at present we treat these all as arbitrary values with `ObjectStructInfo`. Note that, for now, strings and TIR datatypes are also treated as opaque objects. Another noteworthy value in this category is the _null object_ (the result of returning a null pointer in C++ or passing in `None` through the Python FFI), which is returned by the `null_value()` operator.

## Representation of Values at Run Time

Because Relax supports calls to arbitrary `PackedFunc`s that can operate on a low level, it is necessary to define a convention for how values will be represented at run time. At this time, the specification does not require any specific representation and permits compiler implementations to choose their own representations, provided that each value type listed above can be recognized at run time (for dynamic `StructInfo` checks). This means that Relax programs that call `PackedFunc`s directly are not portable across compiler implementations: The `PackedFunc`s used must be able to operate on the run-time representations of values.

Possible specification in terms of the TVM object system:

- Tensors are represented at run time as `NDArray`s (see `include/tvm/NDArray.h`).
- Tuples are represented using TVM `Array`s (in contrast to `NDArray`s), which are immutable (see `include/tvm/runtime/container/array.h`).
- At run time, closures are represented as a `ClosureObj` (see `include/tvm/runtime/container/closure.h`); in the Relax VM these more specifically use the `VMClosureObj` (see [`https://github.com/tlc-pack/relax/blob/relax/include/tvm/runtime/relax_vm/executable.h`](https://github.com/tlc-pack/relax/blob/relax/include/tvm/runtime/relax_vm/executable.h)).
- Shape values are represented at run time as a `ShapeTuple` (see `include/tvm/runtime/container/shape_tuple.h`).
- Strings are represented using TVM's `String` container (see `include/tvm/runtime/container/string.h`).
- We require objects other than the above values used by and returned by `PackedFunc` to inherit from TVM's `Object` class (defined in `include/tvm/runtime/Object.h`). Note that `PackedFunc`s are capable of using and returning all TVM POD (plain-old data) values (see `include/tvm/runtimes/packed_func.h`), which includes some representations that do not inherit from `Object`. In the future, we may define semantics for other values, but at present, these are *unsupported* in Relax and we make no guarantees about the semantics of calling `PackedFunc`s that use or return anything that does not inherit from `Object`.

# Variable Scoping

There are four relevant scopes in Relax, which determine where variables are visible and can be used:

1. Global: `GlobalVar`s can be referenced from any function in the `IRModule`, whether a Relax function or a TIR `PrimFunc`. All global functions are visible to each other and to themselves, allowing for mutual recursion.
2. Function: The parameters to a function (ordinary `Var` nodes) can be referenced from anywhere in that function. In a recursive binding (a `Binding` node where the RHS is a `Function` node or `GlobalVar` being mapped to a function at the `IRModule` level), the variable being bound is also scoped to that function, allowing for defining a recursive function.
3. `SeqExpr`: `Var` nodes defined in a `BindingBlock` in a `SeqExpr` node can be referenced in any later binding within the same `BindingBlock`, in any binding within any later `BindingBlock` in that `SeqExpr` node, or in the `SeqExpr`'s body expression. The variables defined in the `BindingBlock`s leave scope once the `SeqExpr` returns.
4. `DataflowBlock`: `DataflowVar`s introduced in a `DataflowBlock` can be referenced in any later binding within that `DataflowBlock`, but leave scope *once that `DataflowBlock` finishes executing*. Definitions in a `DataflowBlock` that are intended to leave the `DataflowBlock` should be bound to an ordinary `Var`.

Note that Relax variables must be bound _exactly_ once. A global variable is bound if it is mapped to a function in the `IRModule` and a local variable is bound if it appears as a function parameter or if it appears on the left-hand side (LHS) of a binding (`VarBinding` or `MatchCast`).

«If there is another binding to a local variable with the same name as an already-bound variable, that is binding is considered to _shadow_ the previous binding, i.e., it is a binding to a new, distinct variable that happens to have the same name as the existing variable. The new, shadowing variable will exist only in the current scope; if the older variable was defined in an outer scope, then future uses of that name will refer to the older variable. [See the Wikipedia page for more information on variable shadowing.](https://en.wikipedia.org/wiki/Variable_shadowing)»

Below is an example of shadowing, in pseudocode:

```python
@R.function
def func(x: Tensor) -> Tensor:
    if True:
        # the true branch will be a nested SeqExpr and hence a new scope
        # this x will shadow the function parameter x
        x = R.const(1)
        R.print(x) # prints 1
        # the inner x goes out of scope
    else:
        R.print("not executed")
    R.print(x) # this x is the function parameter
    return x
```

# Normal Form

To simplify the writing of Relax passes, we define a normal form for Relax programs, based on the [administrative normal form](https://en.wikipedia.org/wiki/A-normal_form) (A-normal form, or ANF). See [this post](https://matt.might.net/articles/a-normalization/) by Matt Might for a discussion of some of the advantages of ANF in traditional compilation; in particular, ANF results in programs without nesting, which is very convenient for writing program transformations. Because the `StructInfo`-checking rules for operators rely on macros (`FInferShapeInfo`), _this means that the structure of the program can affect `StructInfo` inference_. Putting programs into normal form (and lacking nesting) not only simplifies the writing of these macros but it also ensures that these `StructInfo`-checking rules will be predictable, hence _it is required to transform programs into normal form_ before applying `StructInfo` checking.

The normal form for Relax is very similar to ANF; differences will be noted. Here are the criteria required for a program to be in normal form:
1. Within a `SeqExpr`, the right-hand side of any binding (the `value` field in the AST) must either be a "leaf expression" or a non-leaf expression where all subexpressions are leaf expressions. Leaf expressions are the following: Variables (`Var`, `DataflowVar`, or `GlobalVar`), `Constant`, `ShapeExpr`, `PrimValue`, `StringImm`, `DataTypeImm`, or (_unlike_ ANF) `Tuple`. `Tuple` nodes are considered "leaf" expressions even though they contain nesting purely for convenience in writing passes; many operators rely on grouping arguments using tuples, so that is a form of nesting permitted and expected. Otherwise, non-leaf expressions used as subexpressions must be bound to variables; this includes any non-leaf expressions nested inside a `Tuple`.
2. `SeqExpr`s may appear only in the following locations:
    1. In the `body` field of a `Function` node.
    2. In the `true_branch` and `false_branch` fields of `If` nodes.
3. In fact, the `body` field of a `Function` node and the `true_branch` and `false_branch` fields of `If` nodes _must_ be `SeqExpr`s. If these fields are not `SeqExpr`s, they must be "wrapped" in a `SeqExpr`.
4. Within a `SeqExpr`, `BindingBlock`s must be consolidated. For example, if there is a `BindingBlock` that comes after another `BindingBlock`, the two blocks should be combined to form a single `BindingBlock` with all the bindings in the same order. Consecutive `DataflowBlock`s should be consolidated as well. Empty `BindingBlock`s should be dropped. However, a `DataflowBlock` cannot be consolidated with an ordinary `BindingBlock`. If all the `BindingBlock`s are empty, then the `blocks` field of the `SeqExpr` should be set to an empty list.
5. Calls to `Op` nodes can have custom normalization rules in order to ensure that calls to those operators will conform to certain specific rules (ideally, these should be _more_ and not _less_ restrictive than the other rules of normal form). In particular, `call_tir` and related operators include a custom normalization rule that requires the arguments to the `PrimFunc` to be provided as a tuple _literal_, rather than, say, a variable that evaluates to a tuple.

Programs that are parsed should be "normalized" before performing `StructInfo` checking or before doing any further optimizations. Note that the process of "flattening" `SeqExpr`s and consolidating `BindingBlock`s does increase the visibility of the variables in those `SeqExpr`s and `BindingBlock`s, but this is safe, since it will not cause any variable to be referenced outside of its original scope. The specification does not require any particular method of normalizing a program so long as the final program conforms to the above-listed criteria. Here is a general approach:
1. For each function in the `IRModule`, ensure that the body is a `SeqExpr`. If the body is not a `SeqExpr`, wrap the function body in a `SeqExpr`, creating a new `BindingBlock` to hold `VarBinding`s for any non-leaf expressions that need to be bound to variables.
2. If the function body is already a `SeqExpr`, consolidate all `BindingBlock`s, then check if the `body` field of the `SeqExpr` is a leaf expression. If not, bind it to a new var in the final `BindingBlock` and replace the `SeqExpr` body with the new var.
3. If the function body is not a `SeqExpr`, then recurse down the body's AST, binding any nested non-leaf expressions to a var in the current scope (doing this process in breadth-first order from left to right will respect the evaluation order in the semantics). If the body itself is a non-leaf expression, finally bind it to a var and have the final `SeqExpr` return the new var.
4. If an `If` node is encountered, ensure the `true_branch` and `false_branch` fields are `SeqExpr`s (consolidate `BindingBlock`s if necessary) or "wrap" them in `SeqExpr`s in the same manner as the function body.
5. If a `SeqExpr` node is encountered as the `value` node in a binding, "flatten" the `SeqExpr` by adding its bindings to the current scope and replacing the `SeqExpr` with its body. If the `SeqExpr` body is a non-leaf expression, normalize it recursively in the same manner as in step 3 before replacing the binding. Note that if the current scope (the location of the binding) is a `DataflowBlock` and the nested `SeqExpr` contains an ordinary `BindingBlock`, that indicates a malformed program.
6. For calls to `Op`s, check if it has a custom normalization rule and apply the custom normalization rule.


# Well-Formedness Criteria

Prior to `StructInfo` checking, Relax programs must conform to certain syntactic criteria to be valid, which includes conforming to the expectations of the above-described normal form.

The following criteria apply to all programs (including before normalization):
1. `DataflowVar`s can be bound only inside `DataflowBlock`s. Additionally, a `DataflowVar` may not be used outside of the `DataflowBlock` in which it is defined.
2. A `Var` of any kind used in the program must be either a function parameter or appear on the LHS of a binding exactly once. In the binding where a `Var` is defined, the same `Var` is permitted to occur in the RHS of the binding only if the binding is defining a function (i.e., local functions are permitted to be recursive).
3. A `Var` of any kind may not appear before it is bound. Namely, if a `Var` is bound in a `BindingBlock` in a `SeqExpr`, that `Var` may not appear in bindings that precede the one where it appears on the LHS.
4. «A return structural annotation for a function is not allowed to use any shape variables that are not in scope at the function definition. That is, the only shape variables that can appear on the return structural annotation are those defined in the outer scope or those introduced in the argument structural annotations.»
5. In each function, `PrimExpr` variables (shape variables) similarly may not appear in `ShapeExpr`s or shape annotations before the shape variables are bound (either in function signatures or `MatchCast` bindings). A shape variable is bound only when it appears in a dimension by itself (for example, a dimension consisting of `x` will bind `x`; however, `2*x` is not a binding and is considered an error if `x` has not yet been bound) in a `MatchCast` node or a function argument shape annotation.
6. In a function signature, every shape variable must appear in a binding position at least once; however, (for convenience) we do not enforce any ordering amongst the function arguments—for example, it is permitted to have a shape `x * y` in the first argument and have `x` and `y` appear in binding positions in later arguments. In such a case, the dimensions corresponding to the binding positions will be checked first, allowing the variables to be bound. Then the other dimensions will be checked.
7. The following constructs are not permitted to occur inside `DataflowBlock`s, which must be side effect– and control flow–free: 
    1. Recursive calls to the current function
    2. Calls to a global function that is mutually recursive with the current function
    3. `If` nodes
    
    Calls to Relax functions, `ExternFuncs`, or `Op`s that are not pure are also not permitted, but this must be detected during `StructInfo` checking.
    
8. «For functions that contain recursive calls to themselves or mutually recursive global functions (i.e., those where function `a` calls function `b` and function `b` calls function `a`), a return structural annotation is *required*.»
9. `Op` nodes may appear only as the `op` argument to `Call` nodes. 
10. If a variable has a `StructInfo` annotation, the `ndim` of any `TensorStructInfo` and `ShapeStructInfo`s must match the number of dimensions in their `shape` and `values` fields, respectively.
11. A function definition inside a `DataflowBlock` may not use `DataflowVar`s from the outer scope in its body. We do not define closure capturing for `DataflowVar`s.
12. «At least one global function in the `IRModule` must be externally linked (have a `global_symbol` attribute) in order to serve as a program entry point.»
13. «If a global function has a defined `global_symbol` attribute, the `global_symbol` name must be the same as the `GlobalVar`'s name hint.»
14. If the `shape` field of a `TensorStructInfo` in any structural annotation is given, the only permissible expressions are `Var` (the variable must be in scope at the location of the annotation) or `ShapeExpr` (in which any shape variables used must already be in scope, unless the `TensorStructInfo` is the `struct_info` field of a `MatchCast`, in which case a new shape variable is allowed to appear in a dimension by itself). Additionally, if the `shape` field is a `ShapeExpr`, the number of dimensions must match the `ndim` field.
15. If the `values` field of a `ShapeStructInfo` in any structural annotation is given, any shape variables used in it must already be in scope, unless the `ShapeStructInfo` is the `struct_info` field of a `MatchCast`, in which case a new shape variable is allowed to appear by itself as a member of `values`. Additionally, the `ndim` field must match the length of `values`. 
16. Similarly, if the `value` field of `PrimStructInfo` is defined, any shape variables used in it must already be in scope, unless the `PrimStructInfo` is the `struct_info` field of a `MatchCast`, in which case a new shape variable is allowed to appear by itself as `value`.
17. The `params` and `derive_func` field may not be simultaneously defined in a `FuncStructInfo` annotation; that is, if one is defined, the other must not be defined. Additionally, at least one of `params` and `derive_func` _must_ be defined for each `FuncStructInfo` in an annotation.
18. `PrimValue` nodes are intended only to be used with `value`s consisting of TIR `IntImm`s and `FloatImm`s (with `lanes` set to 1).
19. `PrimStructInfo` annotations should use only the `Int`, `UInt`, or `Float` datatypes for their `dtype` fields.
20. Per [the notes on `DataType`](#notes-on-datatype-and-related-terminology), any `DataType` annotation must have a `lanes` value of 1 for the `Int`, `UInt`, or `Float` datatypes and a `lanes` value of 0 for the `Handle` (`Void`) datatype. Additionally, `bits` must be 64 for `Void`. The supported bitwidths for `Int` and `UInt` are 1, 8, 16, 32, and 64; the supported bitwidths for `Float` are 16, 32, and 64.
21. If a `Function` `f` has an `attrs` field that includes the attribute `relax.force_pure`, `f`'s `is_pure` field must be set to `True`.
22. For `PrimStructInfo`, if the `value` field is defined, the TIR `dtype` for the `PrimExpr` must match the `PrimStructInfo`'s `dtype` field (i.e., the datatypes must be consistent).
23. For any `Call` node where the callee (`op` field) is an `Op` node, if the `Op` has a custom normalization rule, the call must conform to that rule. In particular, applying to the normalization rule to the `Call` should not require any further changes.
24. «All `VDevice`s reference in `StructInfo` annotations _must_ appear in the `IRModule`'s `global_info` map. (Corollary: If no `VDevice` is given in `global_info`, then _all_ `vdevice` fields in `TensorStructInfo` annotations must remain undefined.)»

Additionally, the criteria for normal form listed in [the previous section](#normal-form) must apply to any program that has been normalized.

# Structural Information (`StructInfo`) in Relax

Structural information in Relax is intended to enforce basic guarantees that values are passed correctly between expressions, while also analyzing more complex properties like tensor shapes in a _"best-effort"_ fashion. Namely, anything that cannot be proved statically can instead be checked at run time. Each Relax expression has structural information associated with it. The best-effort nature of the structural system in Relax means that the analysis may detect _some_ errors at compile time and report them, but it may give warnings when it _cannot_ draw conclusions, perhaps suggesting that dynamic checks via `MatchCast` should be inserted. Note that the precision of the static analysis can potentially be improved by some compile-time optimizations like constant propagation, function inlining, and other partial evaluation–like transformations.

Tensor shapes are the primary motivation for including structural information in Relax, as shape information is particularly important for memory planning. In Relay, shapes are part of tensor types and there is much analysis of tensor shapes done at compile time. While this allows Relay's type system to make strong guarantees about tensor shapes, it results in greater complexity in type checking and makes it difficult to implement new operators or handle cases like tensors with symbolic shapes. By contrast, Relax's `StructInfo` system uses expressions to encode tensor shapes, which allows for using shape variables and arithmetic expressions to encode a rich variety of shape constraints. Note, however, that the structural system could potentially be extended to encode and analyze further information, like tensor sparsity or density.

## Defining Structural Information

The structural information in Relax corresponds to the values in the language:
* `TensorStructInfo` describes tensor values. The `dtype` field gives the datatype (with `Void` indicating a statically unknown datatype), the `ndim` field gives the rank (with -1 indicating a statically unknown rank). Unlike `DynTensorType`, there is an optional `shape` field which, if defined, describes the shape of the tensor using either a `ShapeExpr` or a `Var` (with `ShapeStructInfo`). If `shape` is a `ShapeExpr`, the `PrimExpr`s in the `ShapeExpr`'s dimensions describe how to compute each dimension of the shape (or are constants). If `shape` is a `Var`, the `Var` can assign the result of an arbitrary computation that returns a shape value, which can be useful for memory planning. The `vdevice` field, if present, indicates on which device the tensor value is located, since `NDArray`s can be allocated on different devices (if absent, then that means that the device is unspecified).
* `ShapeStructInfo` describes shape values. It has an `ndim` field that gives the number of dimensions in the shape (with -1 indicating that it is statically unknown). It additionally has an optional `values` field. If defined, `values` gives a list of `PrimExpr`s that indicate how to compute the dimensions of the shape, potentially providing further information for static analyses.
* `PrimStructInfo` describes `PrimValue`s, giving their TIR datatype.
* `TupleStructInfo` describes tuple values, namely by giving the `StructInfo` for each of the tuple's members via `fields`.
* `FuncStructInfo` describes closure values or `PackedFunc`s. There are two ways in which to specify `FuncStructInfo`:
    1. By specifying `params` and `ret` (for closures). `params` gives the `StructInfo` corresponding to each of the function's parameters and `ret` gives the `StructInfo` corresponding to the result of calling the function. In this case, the `derive_func` field is left undefined.
    2. By giving a `derive_func` macro (for `PackedFunc`s). The `derive_func` macro is takes a call to the corresponding `PackedFunc` and the variable mapping context and returns the `StructInfo` of the result. In this case, the `params` field is left undefined and the `ret` field is ignored.
* `ObjectStructInfo` describes arbitrary object values.

### Expressing Shape Dimensions

A tensor shape is a tuple of TIR `PrimExpr`s, where each `PrimExpr` corresponds to a dimension. The use of TIR `PrimExpr`s for shape dimensions allows shape computations to express complex constraints that include variables and integer arithmetic expressions in addition to just constant dimensions.

**Scope of Shape Variables**

New shape variables can be bound in two places in a Relax program: In `TensorStructInfo` or `ShapeStructInfo` annotations on function parameters or as the `struct_info` parameter in a `MatchCast` binding. Shape variables used in the function signature are scoped to the entire function in which they appear (including in the return structural annotation). Shape variables used in `MatchCast` bindings are scoped only to the `SeqExpr` in which they appear.

**Informal Semantics of `PrimExpr`s for Dimensions**

1. Shape variables can be bound to a value exactly once, either at the start of a function for shape annotations on function arguments or in `MatchCast` bindings. In particular, matching a `PrimExpr` consisting only of an uninitialized shape variable is treated as its binding (see below on `MatchCast`). After a shape variable has been bound for the first time, future uses of it will refer to the same value.
2. It is not legal to use a shape var that has not yet been bound. This results in an error at compile time.
3. «Local functions will "capture" defined shape variables from the parent scope with their present values in the resulting closure.»
4. If all variables in the `PrimExpr` are defined, `PrimExpr` arithmetic will generally be evaluated according to the semantics of TIR.

### Evaluating `MatchCast`

Because structural information is checked in a "best-effort" fashion, it is not always possible for the compiler to statically draw conclusions about all details of a given value's structural information. Hence, `MatchCast` allows for checking this information at run time, similar to a typecast. However, `MatchCast` also allows for binding shape variables in the process of pattern matching, hence the "match" portion of its name.

This section describes the run-time checking performed by `MatchCast(var, value, struct_info)`, for each combination of value and structural annotation (if `var` is defined, then `value` will be bound to `var` as discussed in the [general section on semantics](#detailed-semantics)). If any check given below fails, an error is raised by the `MatchCast`.

1. If `struct_info` is `ObjectStructInfo`, then no additional check is performed. All values in Relax are objects.
2. If `struct_info` is `TensorStructInfo(shape, dtype, vdevice, ndim)`, then check that `value` is a tensor value, that it has a rank of `ndim` (if `ndim` is not -1) and a datatype of `dtype` (if `dtype` is not `Void`), and is located on the device denoted by `vdevice` (if defined). If `shape` is defined, consider the following cases:
    1. If `shape` is a `Var`, then check that the concrete shape of `value` matches the value bound to the `Var`.
    2. If `shape` is a `ShapeExpr`, then compare the fields of the `ShapeExpr` to the concrete shape of `value`, dimension by dimension (comparing the `i`th field of the `ShapeExpr` to the `i`th dimension of the shape of `value`). Give an error if the number of the dimensions does not match the number of fields in the `ShapeExpr`.
        1. If a field of the `ShapeExpr` consists of only an unbound shape variable, then bind that variable to the value of the dimension.
        2. Otherwise, evaluate the field of the `ShapeExpr` and ensure that it matches the concrete value of the dimension.
3. If `struct_info` is `PrimStructInfo(dtype, v)`, then check that `value` is a `PrimValue` and that the underlying scalar has datatype `dtype` in TIR (according to TIR's type-checking rules). If `v` is defined, then check that `value` and `v` match numerically.
4. If `struct_info` is `ShapeStructInfo(ndim, values)`, then check that `value` is a shape value, that it has `ndim` dimensions (if `ndim` is not -1). If `values` is defined, then compare it to the concrete shape value (comparing the `i`th member of `values` to the `i`th field of the shape value):
    1. If the `i`th member of `values` consists of only an unbound shape variable, then bind that variable to the `i`th field of the the concrete shape value.
    2. Otherwise, evaluate the `i`th member of `values` and check that it is equal to teh `i`th field of the concrete shape value.
5. If `struct_info` is `TupleStructInfo(fields)`, then check that `value` is a tuple value with `n` fields, where `n` is the length of `fields`. Also recursively check the `i`th field of the tuple value against the `i`th member of `fields`.
6. If `struct_info` is `FuncStructInfo(params, ret, purity, derive_func)`, then if `params` is defined, check that `value` is a closure value; if `derive_func` is defined, check that `value` is a `PackedFunc`. No further validation may be done on a `PackedFunc`. «If `value` is a closure value, then it can contain run-time structural information indicating its purity and the structural information of its intended arguments and return value that can be compared against `purity`, `params`, and `ret`.»

### Checking Structural Information at the Start and End of a Function

Shape variables are bound at the start and end of a function or in `MatchCast` bindings. This checking is done similarly to `MatchCast`, though with a slight difference: per rule #6 in under the [well-formedness criteria](#well-formedness-criteria), we allow shape variables to appear in arguments in any order so long as shape variables appear in a binding position at least once. This requires us to check the shapes of arguments dimension by dimension in a specific order.

Suppose a function has the following signature, where the `Si` are structural annotations:

```python
def f(arg1 : S1, arg2 : S2, ..., argn : Sn) -> Sr: 
    return body
```

The dimensions corresponding to variables in binding positions are checked first. A binding position is when a shape variable appears by itself in a dimension (a field in `values` in `ShapeStructInfo`, the `value` field of `PrimStructInfo`, or a field in `shape` for `TensorStructInfo`). For each variable in a binding position that appears among the `Si`, we check the corresponding field of the concrete value in order to assign to it a value (recursing down the structure if necessary).

1. For `PrimStructInfo`, we compare `value` to the concrete primitive value.
2. For `ShapeStructInfo`, we compare the field within `values` to the corresponding member of the shape.
3. For `TensorStructInfo`, we compare the field of `shape` to the length of the corresponding dimension of the tensor.
4. For other `StructInfo`, it may be necessary to match values recursively. In the case of a closure, the runtime shape information will be needed.

Having found a value, that value is bound to the shape variable. The rest of the tensor shapes and the return value can then be checked from left to right per the following macro:

```python
def f(arg1, arg2, ..., argn):
    MatchCast(arg1, S1)
    MatchCast(arg2, S2)
    ...
    MatchCast(argn, Sn)
    ret_var = body
    MatchCast(ret_var, Sr)
    return ret_var
```

For a concrete example, suppose that the function has a signature
`def f(x: R.Tensor([M * N]), y: R.Tensor([M, N])) -> R.Tensor([N * N, M * M]): ...`.
In this case, `M` would first be bound by checking dimension 0 of the value of `y`, `N` would then be bound by checking dimension 1 of the value of `y`. Next, the shape of `x` would then be compared against `M * N` using the bound values, then the shape of `y` would be compared against `(M, N)` using the bound values. At the end of the function, the shape of the return value would be compared against `(N * M, M * M)` using the bound values.

### Invariants for `TensorStructInfo`

Because the `shape` field of `TensorStructInfo` is an expression (either a `Var` or `ShapeExpr`), that expression may have its own `StructInfo`. In any `TensorStructInfo` derived by the below inference rules for `StructInfo` or in any `StructInfo` annotation, the following properties must hold of the `shape` field in `TensorStructInfo`:
1. If the `shape` field is a `Var`, the `Var` must have `ShapeStructInfo`. The `ndim` for the `Var`'s `ShapeStructInfo` must match that of the `TensorStructInfo`.
2. If the `shape` field is a literal `ShapeExpr`, then the `ndim` for the `TensorStructInfo` must match the number of fields in the `shape`'s `values` field (this is noted in the [well-formedness rules](#well-formedness-criteria)).

Any shape variables that appear in the `ShapeStructInfo` must be in scope where the annotation appears.

In particular, it is not permitted for the `TensorStructInfo` to have an unknown rank (`ndim` of -1) when the `shape` field has a non-negative `ndim`.

## Subtyping for `StructInfo`

Relax implements subtyping for `StructInfo`, which means that values with some `StructInfo` can be accepted where values with more general `StructInfo` are accepted We will denote the subtyping relationship as `S1 <: S2`, indicating that `S1` is a subtype of `S2`. For example. if `S1 <: S2` and some function expects an argument with `StructInfo` `S2`, then passing a value with `StructInfo` `S1` to that function is permitted; passing a value with `StructInfo` `S2` as an argument to a function that expects `S1` for that argument is *not* permitted—the value would have to be dynamically cast to `S1` using `MatchCast`.

Note that judging subtyping requires potentially reasoning about arbitrary `ShapeExpr`s. We assume that the compiler is able to draw the following three conclusions about two shape expressions, acting conservatively (it will consider values to be _definitely_ equal or _definitely not_ equal only if it is certain):
* They are _definitely_ statically equal in all cases.
* They are _possibly_ statically equal.
* They are _definitely not_ statically equal in at least one case.

1. Reflexivity: `S1 <: S1` for all `S1`.
2. Transitivity: For all `S1`, `S2`, and `S3`, if `S1 <: S2` and `S2 <: S3`, then `S1 <: S3`.
3. For all `S1`, `S1 <: ObjectStructInfo()`.
4. For `TensorStructInfo`:
    1. Given any datatype `d`, an arbitrary `ndim` `n`, an arbitrary expression `s` (possibly undefined), and an arbitrary `VDevice` `v` (possibly undefined), `TensorStructInfo(ndim=n, dtype=d, shape=s, vdevice=v) <: TensorStructInfo(ndim=-1, dtype=d, vdevice=v)`.
    2. Given any datatype `d`, an arbitrary `ndim` `n`, an arbitrary expression `s` (possibly undefined), and an arbitrary `VDevice` `v` (possibly undefined), `TensorStructInfo(ndim=n, dtype=d, shape=s, vdevice=v) <: TensorStructInfo(ndim=n, dtype=Void(), shape=s, vdevice=v)`.
    3. Given any datatype `d`, an arbitrary `ndim` `n`, an arbitrary `VDevice` `v` (possibly undefined), and an arbitrary expression `s`, `TensorStructInfo(ndim=n, dtype=d, vdevice=v, shape=s) <: TensorStructInfo(ndim=n, dtype=d, vdevice=v, shape=undefined)`.
    4. Given any datatype `d`, an arbitrary `ndim` `n`, an arbitrary `VDevice` `v` (possibly undefined), and arbitrary expressions `s1` and `s2` (both defined), then `TensorStructInfo(ndim=n, dtype=d, vdevice=v, shape=s1) <: TensorStructInfo(ndim=n, dtype=d, vdevice=v, shape=s2)` if `s1` and `s2` are _definitely_ statically equal. We say that `TensorStructInfo(ndim=n, dtype=d, vdevice=v, shape=s1) <: TensorStructInfo(ndim=n, dtype=d, vdevice=v, shape=s2)` _possibly_ holds if `s1` and `s2` are _possibly_ statically equal.
    5. Given any `VDevice` `v` (that is defined), any datatype `d`, an arbitrary `ndim` `n`, and an arbitrary expression `s` (possibly undefined), then `TensorStructInfo(ndim=n, dtype=d, vdevice=v, shape=s) <: TensorStructInfo(ndim=n, dtype=d, vdevice=undefined, shape=s)`.
5. For `ShapeStructInfo`:
    1. Given an arbitrary `ndim` `n` and an arbitrary set of values `v` (possibly undefined), `ShapeStructInfo(ndim=n, values=v) <: ShapeStructInfo(ndim=-1)`.
    2. Given an arbitrary `ndim` `n` and an arbitrary set of values `v` (not undefined), `ShapeStructInfo(ndim=n, values=v) <: ShapeStructInfo(ndim=n, values=undefined)`.
    3. Given an arbitrary `ndim` `n` and two arbitrary sets of values `v1` and `v2` (both defined), `ShapeStructInfo(ndim=n, values=v1) <: ShapeStructInfo(ndim=n, values=v2)` if, for all valid `i`, `v1[i]` and `v2[i]` can be proven to be _definitely_ statically equal. We say that `ShapeStructInfo(ndim=n, values=v1) <: ShapeStructInfo(ndim=n, values=v2)` _possibly_ holds if `v1` and `v2` are _possibly_ statically equal.
6. Given two lists of `StructInfo` `fields1` and `fields2`, `TupleStructInfo(fields=fields1) <: TupleStructInfo(fields=fields2)` if `fields1` and `fields2` are the same length and for all `i`, `fields1[i] <: fields2[i]`. We consider the subtyping relationship to _possibly_ hold if any of the subtyping relationships for the fields only possibly holds.
7. For `PrimStructInfo`:
    1. `PrimStructInfo(dtype=dt1) <: PrimStructInfo(dtype=dt2)` (where the `value` field is undefined for both) holds if `dt1` and `dt2` are the same. That is, we do not have subtyping for TIR datatypes.
    2. Let `dt` be a datatype. `PrimStructInfo(dtype=dt, value=v) <: PrimStructInfo(dtype=dt)` for any `PrimExpr` `v`; that is, if the `value` field is undefined for a `PrimStructInfo`, then it is a supertype to a `PrimStructInfo` with a defined `value` field.
    3. Let `dt` be a datatype. `PrimStructInfo(dtype=dt, value=v1) <: PrimStructInfo(dtype=dt, value=v2)` _definitely_ holds if `v1` and `v2` can be proven to be statically equal. The relation _possibly_ holds if `v1` and `v2` are _possibly_ statically equal.
8. For `FuncStructInfo`:
    1. Given an arbitrary derivation function `derive_func`, `FuncStructInfo(ret=ObjectStructInfo(), derive_func=derive_func) <: FuncStructInfo(ret=ObjectStructInfo(), derive_func=empty_derive)`.
    2. Corollary, following from reflexivity: For two `FuncStructInfo` `F1` and `F2` with undefined `params`, `F1 <: F2` only if `F1.derive_func` and `F2.derive_func` are identical.
    3. Given a list of `StructInfo` parameters `P` and a `StructInfo` return annotation `R`, then `FuncStructInfo(params=P, ret=R, purity=True) <: FuncStructInfo(params=P, ret=R, purity=False)`. That is, a pure function can be passed where an impure one is accepted, but not vice versa.
    3. Given two lists of `StructInfo` parameters `P1` and `P2`, two `StructInfo` annotations `R1` and `R2`, and a Boolean `purity`, `FuncStructInfo(params=P1, ret=R1, purity=purity) <: FuncStructInfo(params=P2, ret=R2, purity=purity)` if `P1` and `P2` are the same length and for all `i`, `P2[i] <: P1[i]` and `R1 <: R2`. We consider the subtyping relationship to _possibly_ hold if any of the subtyping relationships given only possibly holds.

These rules allow us to define the least upper bound (LUB) for any two `StructInfo` `S1` and `S2`, meaning that it is the most specific `StructInfo` `S` for which `S1 <: S` and `S2 <: S` ("most specific" meaning that if there exists some other `S'` for which `S1 <: S'` and `S2 <: S'`, then `S <: S'`), modulo reasoning about arithmetic (for example, the compiler may judge that two shape expressions are _possibly_ equivalent rather than _definitely_ equivalent). The LUB is guaranteed to exist for any two `StructInfo` because all `StructInfo` are subtypes of `ObjectStructInfo`.

We can define how to find the LUB of two structural information annotations (modulo arithmetic reasoning) as follows, in pseudocode:

```python
def unify_struct_info(S1: StructInfo, S2: StructInfo) -> StructInfo:
    if S2 is ObjectStructInfo:
        return S1
    if S1 is ObjectStructInfo:
        return S2
    if S1 and S2 do not match types (e.g., not both TensorStructInfo, etc):
        return ObjectStructInfo()
    if S1 and S2 are both PrimStructInfo:
        if S1.dtype != S2.dtype:
            return ObjectStructInfo()
        if S1.value or S2.value is undefined:
            return PrimStructInfo(dtype=S1.dtype, value=undefined)
        if S1.value can be statically proven to match S2.value:
            return S1
        # values either proven not to match or unknown
        return PrimStructInfo(dtype=S1.dtype, value=undefined)
    if S1 and S2 are both ShapeStructInfo:
        if S1.ndim == -1:
            return S1
        if S2.ndim == -1:
            return S2
        if S1.ndim != S2.ndim:
            return ShapeStructInfo(ndim=-1)
        if S1.ndim == S2.ndim:
            if S1.values is undefined:
                return S1
            if S2.values is defined:
                return S2
            if S1.values can be statically proven to match S2.values:
                return S1
            # values either proven not to match or unknown
            return ShapeStructInfo(ndim=S1.ndim) # leave values undefined
    if S1 and S2 are both TensorStructInfo:
        ndim = S1.ndim if S1.ndim == S2.ndim else -1
        dtype = S1.dtype if S1.dtype == S2.dtype else Void
        vdev = S1.vdevice if S1.vdevice == S2.vdevice else undefined
        if (
                S1.ndim == -1 or S2.ndim == -1 or S1.ndim != S2.ndim 
                or S1.shape is undefined or S2.shape is undefined
            ):
            return TensorStructInfo(ndim=ndim, dtype=dtype) # leave shape undefined
        # both shapes are defined
        if S1.shape can be proven to equal S2.shape:
            return S1
        # either proven to be unequal or cannot be concluded whether they are equal
        return TensorStructInfo(ndim=ndim, dtype=dtype, vdevice=vdev, shape=undefined)
    if S1 and S2 are both TupleStructInfo:
        if S1.fields and S2.fields are of different lengths:
            return ObjectStructInfo()
        return TupleStructInfo(
            unify_struct_info(S1.fields[i], S2.fields[i]) 
            for 0 <= i < length of S1.fields
        ])
    if S1 and S2 are both FuncStructInfo:
        if S1.params and S2.params are not both defined or both undefined:
            return ObjectStructInfo()
        if S1.params and S2.params are both undefined:
            # they must be the same function, not bothering to check eta-equivalence
            if S1.derive_func == S2.derive_func:
                return S1
            return FuncStructInfo(ret=ObjectStructInfo(), derive_func=empty_derive)
        if S1.params and S2.params are both defined:
            if S1.params and S2.params do not have the same length:
                return ObjectStructInfo()
            # the LUB is pure if they're both pure and false if either isn't
            purity = S1.purity and S2.purity
            unified_params = []
            for 0 <= i < length of S1.params:
                unified_param = unify_struct_info(S1.params[i], S2.params[i])
                # That is, if the params judged to be equal, use them. 
                # If there is some pair that is not equal, 
                #   we can't unify these types except with ObjectStructInfo.
                # This rule should suffice in practice; otherwise we would 
                # need to give a full definition of the GLB
                if unified_param <: S1.params[i] and unified_param <: S2.params[i]:
                    unified_params[i] = unified_param
                else:
                    return ObjectStructInfo()
            return FuncStructInfo(params=unified_params, ret=unify_struct_info(S1.ret, S2.ret), purity=purity)
```

## Deriving the Structural Information for Each Expression

For each kind of expression, we can recursively build up the structural information associated with the expression.

### Checking Purity

The below derivation rules will explain in formal detail how Relax checks the correctness of purity annotations and enforces that impure calls are not made inside `DataflowBlock`s. At a high level, it operates by the following principles:
1. Calls to `ExternFunc`s (which thus includes any expression whose `StructInfo` is `FuncStructInfo` with a `derive_func` included) are assumed to be impure by default. The `call_pure_packed` operator can be used to indicate to the compiler that a particular call to an `ExternFunc` is, in fact, pure.
2. `Op` nodes must have an attribute called `FPurity`, which is a boolean flag that indicates whether or not the operator is pure. If the operator can have visible side effects in any case at all, it should be considered impure.
3. For Relax `Function`s, the purity will depend on the `is_pure` annotation (which must be user-supplied).

Thus, the `StructInfo` system can determine whether a call is pure based on the above principles: For operators, it refers to `FPurity` and otherwise it refers to the `FuncStructInfo` (using the `purity` field for functions with `params` defined and assuming that any function with a `derive_func` defined is impure). If any such call occurs inside a `DataflowBlock` or a `Function` whose `is_pure` field is set to `True`, that is treated as a type error.

For verifying the purity of a function, however, there is one workaround permitted: If the function has the `relax.force_pure` attribute mapped to `True` in its `attrs`, then impure calls will be disregarded. This accounts for situations where individual actions may be impure (like mutating a value) but the overall effect of the function is pure (e.g., if the value that is mutated is one that is created inside the function, meaning that no externally-visible memory was ever mutated). This case is unlikely to be common for input programs, though `relax.force_pure` is used frequently in later stages of compilation.

### Auxiliary Procedures

**`derive_func` for `FuncStructInfo`**

There are two special `derive_func` values built into the compiler that are used for checking the structural information of `PackedFunc`s.

The first is `default_derive`, giving a simple way to determine the resulting structural information of a `PackedFunc` from its `StructInfo` arguments. `default_derive` takes one argument that is a `Call` node and is defined as follows: 
1. Suppose its call node argument is `Call(op, [arg1, arg2, ..., argn], sinfo_args=[aS1, aS2, ..., aSn])`.
2. If `sinfo_args` is of length 0, then return `ObjectStructInfo()`.
3. If `sinfo_args` is of length 1, then return `aS1`.
4. If `sinfo_args` is of a greater length than 1, then return `TupleStructInfo(fields=[aS1, aS2, ..., aSn])`.

The second is `empty_derive`, which is the weakest possible derivation. It simply returns `ObjectStructInfo` regardless of its argument. This is used for worst-case deducation of `StructInfo` for a `PackedFunc`.

**Erasing Out-of-Scope Information**

When returning a value from an inner scope to an outer scope (namely, the `body` field of a `SeqExpr`, which may use variables defined in the binding blocks, and the `body` field of a `Function`, which may use variables defined in the function body), it may be possible for the derived `TensorStructInfo` or `ShapeStructInfo` to contain Relax variables or shape vars that have gone out of scope. We defined a procedure to check for any of these out-of-scope variables and weaken the structural information not to include it. The procedure is defined below, in pseudocode:

```python
def erase_to_well_defined(
    s: StructInfo, 
    var_scope: set of Relax vars in current scope, 
    shape_var_scope: set of shape vars in current scope)
    -> StructInfo:

    if s is ObjectStructInfo:
        return s
    if s is PrimStructInfo:
        return s
    if s is TensorStructInfo:
        if s.shape is defined:
            if (s.shape is a Relax var that is not in var_scope
                or s.shape is a ShapeExpr that contains any shape var not in shape_var_scope):
                # leave shape undefined
                return TensorStructInfo(ndim=s.ndim, dtype=s.dtype)
            else:
                return s
        else:
            return s
    if s is ShapeStructInfo:
        if (s.values is defined 
            and any member of s.values contains a shape var not in shape_var_scope):
            # leave values undefined
            return ShapeStructInfo(ndim=s.ndim)
    if s is TupleStructInfo:
        return TupleStructInfo(
            fields=[
                erase_to_well_defined(field, var_scope, shape_var_scope)
                for field in s.fields
            ]
        )
    if s is FuncStructInfo:
        if params is defined:
            new_params = []
            for param in s.params:
                if param contains unbound shape variables:
                    insert unbound shape variables into shape_var_scope
                new_params.append(erase_to_well_defined(param, var_scope, shape_var_scope))
            ret = FuncStructInfo(
                params=new_params,
                ret=erase_to_well_defined(s.ret, var_scope, shape_var_scope)
            )
            remove any unbound shape variables added into shape_var_scope above
            return ret
        else:
            return FuncStructInfo(
                ret=erase_to_well_defined(s.ret, var_scope, shape_var_scope),
                derive_func=s.derive_func
            )
```

**Substituting Free Shape Variables in `FuncStructInfo`**

The `params` field of `FuncStructInfo` can contain free shape variables, indicating that these shape variables are bound to the corresponding dimensions of the argument when the function is called. For checking the compatibility of two function types, we can construct a mapping of shape variables and then substitute shape variables according to the mapping. The mapping can be constructed by doing a simple structural match, as when checking alpha-equivalence.

For clarity, additional detail on how the mapping should be constructed is given here in pseudocode:

```python
def get_shape_var_mapping(S1: StructInfo, S2: StructInfo) -> {tir::Var, PrimExpr}:
    if S1 and S2 are not the same type:
        return {}
    if S1 and S2 are both PrimStructInfo:
        return {}
    if S1 and S2 are both TupleStructInfo:
        if S1.fields and S2.fields don't have the same length:
            return {}
        ret = {}
        for 0 <= i < length of S1.fields:
            ret = union of ret and get_shape_var_mapping(S1.fields[i], S2.fields[i])
        return ret
    if S1 and S2 are both FuncStructInfo:
        if S1 and S2 both have params defined and the params are the same length:
            ret = {}
            for 0 <= i < length of S1.params:
                ret = union of ret and get_shape_var_mapping(S1.params[i], S2.params[i])
            # don't look at the return field; it's not a binding position
            return ret
        else:
            return {}
    if S1 and S2 are both ShapeStructInfo:
        if S1 and S2 both have values defined and the values are the same length:
            ret = {}
            for 0 <= i < length of S1.values:
                if S1.values[i] is an unbound shape variable:
                    ret[S1.values[i]] = S1.values[i]
            return ret
        else:
            return {}
    if S1 and S2 are both TensorStructInfo:
        if (
            S1 and S2 both have shape defined 
            and the shapes are both ShapeExprs 
            and their values fields are the same length
        ):
            ret = {}
            for 0 <= i < length of S1.shape.values:
                if S1.shape.values[i] is an unbound shape variable:
                    ret[S1.shape.values[i]] = S2.shape.values[i]
            return ret
        else:
            return {}
```

**Checking Compatibility**

In many cases during the derivation of structural information, it is important to judge when two distinct structural information encodings are compatible with each other or when they are too different from each other to be reconciled, which can indicate an error. In the case of shape information, this could mean having two symbolic shapes that can be proven not to be equal to each other. Because shape expressions can contain arithmetic and it can be very difficult to statically prove whether two arithmetic expressions are equal, we permit the compiler implementation to make a best-effort attempt to prove equality for arithmetic expressions. (The user can insert a `MatchCast` to check definitively.) Since the checks are best-effort, the compatibility check will only report incompatibility if two values are _definitely_ different from each other.

We can check if some structural information `S1` is accepted where structural information `S2` is expected by the process given below, which we refer to as `check_compability(S1, S2)` for convenience. `check_compatibility` can find that `S1` and `S2` are compatible, possibly compatible, or incompatible. "Incompatible" indicates a definite mismatch that should result in a compiler error; "possibly compatible" indicates that the structures may or may not match and should likely result in a compiler warning (indicating that a user may want to insert a dynamic check). An invariant that should should is that if `check_compatibility(S1, S2)` returns "compatible" or "possible compatible", `erase_struct_info(S1) <: erase_struct_info(S2)` should hold; that is, compatibility of structural information should be consistent with typing rules.

1. If `S2` is `ObjectStructInfo`, then they are compatible.
2. Otherwise, if `S1` and `S2` are not both `TensorStructInfo` or both `TupleStructInfo`, etc. (besides `ObjectStructInfo`), then report an incompatibility.
3. If `S1` and `S2` are both `TupleStructInfo`:
    1. If `S1.fields` is not the same length as `S2.fields`, they are incompatible
    2. Call `check_compability(S1.fields[i], S2.fields[i])` for all `i`. If any pair of fields is incompatible, then `S1` and `S2` are incompatible. If no pair of fields is incompatible but at least one is possibly compatible, then `S1` and `S2` are possibly compatible. If all pairs of fields are compatible, then `S1` and `S2` are compatible.
4. If `S1` and `S2` are both `ShapeStructInfo`:
    1. `S2.ndim` is -1, then they are compatible.
    2. Otherwise, give an error if `S1.ndim` does not match `S2.ndim`. 
    3. If `values` is not defined for `S2`, then they are compatible.
    4. If `values` is defined for `S2` but not defined for `S1`, then they are possibly compatible.
    5. If `values` is defined for both `S1` and `S2`, then the two are incompatible if `S1.values[i]` can be proven to be _not_ equal to `S2.values[i]` for some `i`. If all members can be proven to be equal, then they are compatible. Otherwise, if at least one pair of values cannot be proven to be either equal or unequal, then they are possibly compatible.
5. If `S1` and `S2` are both `PrimStructInfo`:
    1. If `S1.dtype` and `S2.dtype` do not match, then they are incompatible.
    2. If `value` is not defined for `S2`, then they are compatible.
    3. If `value` is defined for `S2` but not for `S1`, then they are possibly compatible.
    4. If `value` is defined for both `S1` and `S2`, then they are compatible if `S1.value` can be statically proven to be equal to `S2.value`. They are possibly compatible if `S1.value` is possibly statically equal to `S2.value` but it cannot be proven. They are incompatible if `S1.value` can be proven to _not_ be statically equal to `S2.value`.
6. If `S1` and `S2` are both `TensorStructInfo`:
    1. If `S2.dtype` is not `Void`, `S1.dtype` is not `Void`, and `S1.dtype` and `S2.dtype` do not match, then they are incompatible.
    2. If `S2.ndim` is not -1, `S1.ndim` is not -1, and `S1.ndim` and `S2.ndim` do not match, then they are incompatible.
    3. If `S2.vdevice` is defined and does not match `S1.vdevice`, then they are incompatible.
    4. If `S2.shape` is not defined, then they are compatible pending step 8.
    5. If `S2.shape` is defined and `S1.shape` is not defined, then they are possibly compatible.
    6. Otherwise, if both `shape` fields are given and either is a `Var`, then consider `S1` and `S2` compatible (pending step 8) if the compiler can statically prove that the `Var` holds the same value as the other `shape` field, consider them possibly compatible if the compiler cannot draw a conclusion one way or the other, and consider them incompatible if the `Var` definitely has a different value from the other `shape`.
    7. If both `shape` fields are given and they are both `ShapeExpr` nodes, then `S1` and `S2` are incompatible if the compiler can prove that some dimension of `S1.shape` is _not_ equal to the corresponding dimension of `S2.shape`. Otherwise, if the all dimensions can be proven to be equal, then consider them compatible pending step 8. If at least one pair of dimensions cannot be proven to be equal or unequal, consider them possibly compatible.
    8. If we have concluded `S1.shape` and `S2.shape` to match in step 4, 6, or 7, then consider `S1` and `S2` possibly compatible if `S1.dtype` is `Void` while `S2.dtype` is not `Void` or if `S1.vdevice` is undefined but `S2.vdevice` is defined. Otherwise, consider `S1` and `S2` compatible.
7. If `S1` and `S2` are both `FuncStructInfo`:
    1. If `S1` and `S2` don't both have defined `params` or both have undefined `params`, consider them incompatible.
    2. If both `S1` and `S2` have undefined `params`, consider them compatible if they have an identical `derive_func` and consider them possibly compatible if they have different `derive_func`s (as they is no further way to introspect the `derive_func` and draw static conslusions about `PackedFunc`s).
    3. If `params` is defined for both `S1` and `S2`:
        1. Consider them incompatible if the `params` have different lengths. 
        2. If the `purity` of `S1` is `False` but the `purity` of `S2` is `True`, then consider them incompatible.
        3. Next, map unbound shape variables as follows: Get a variable mapping `m` by applying `get_shape_var_mapping(S1.params[i], S2.params[i])` for all values of `i`, taking the union of all resulting mappings. Next, substitute all occurrences of the shape variables in `S1` with their values in `m`.
        4. If `check_compatibility(S2.params[i], S1.params[i])` (note the direction of the check: see the subtyping rule for `FuncType`) is incompatible for any `i` or if `check_compatibility(S1.ret, S2.ret)` is incompatible, then they are incompatible. Otherwise, if `check_compatibility(S2.params[i], S1.params[i])` is possibly compatible for any `i` or if `check_compatibility(S1.ret, S2.ret)` is possibly compatible, consider `S1` and `S2` possibly compatible. Consider `S1` and `S2` compatible only if all checks are compatible.

### Derivation Rules

Let `Γ` be the `StructInfo` context for Relax variables and let `Σ` track which shape variables are in scope.

1. «Prepopulate `Γ` with the annotated types of all global functions (see the rule for `Function` nodes) that are called mutually recursively. Afterwards check the structural information of the global functions one at a time and populate the entry of `Γ` corresponding to that `GlobalVar`.»
2. For a variable (`Var`, `DataflowVar`, or `GlobalVar`) `v`, look up `Γ[v]` for the structural information.
3. For `Constant(value)`, the resulting structural information is `TensorStructInfo(ndim, dtype, shape, vdevice=undefined)` where `ndim` is the concrete rank of `value`, `dtype` is the concrete datatype used in `value`, and `shape` is a `ShapeExpr` giving the concrete shape of `value`. For example, for `Constant(1)`, `shape` is `ShapeExpr([])` and for `Constant([1, 2])`, `shape` is `ShapeExpr([IntImm(2, "int64")])`.
4. For `PrimValue(prim_expr)`, the resulting `StructInfo` is `PrimStructInfo(dt)`, where `dt` is the datatype of `prim_expr`, derived according to the type-checking rules for TIR.
5. For `StringImm(s)`, the resulting `StructInfo` is `ObjectStructInfo()`.
6. For `DataTypeImm(dt)`, the resulting `StructInfo` is `ObjectStructInfo()`.
7. For `Tuple(fields)`, suppose that `fields` is comprised of expressions `E1`, `E2`, ..., `En`. Let the `StructInfo` for these expressions be `S1`, `S2`, ..., `Sn`, respectively. Then the resulting `StructInfo` is `TupleStructInfo(fields=[S1, S2, ..., Sn])`.
8. For `ShapeExpr(values)`, the resulting structural information is `ShapeStructInfo(ndim, values)`, where `ndim` is the length of `values`.
9. For `If(cond, true_branch, false_branch)`, we compare the structural information of `true_branch` and `false_branch` (call these `S_t` and `S_f`, respectively). The resulting structural information is `unify_struct_info(S_t, S_f)`.
10. For `SeqExpr(blocks, body)`:
    1. For each binding block in `blocks` (call the current one `block`):
        1. Process each binding in the block, updating `Γ` and `Σ` accordingly (this is discussed in detail below).
        2. If `block` is a `DataflowBlock`, then remove all `DataflowVar`s introduced in `block` from `Γ` before proceeding to the next block.
    2. Next derive the structural information for `body`. Let us call this `S`.
    3. Remove all Relax variables introduced in `blocks` from `Γ` and all shape variables introduced in `blocks` from `Σ`.
    4. The structural information of the entire `SeqExpr` is `erase_to_well_defined(S, Γ, Σ)`.
11. For handling variable bindings:
    1. If `v` is the argument to a function, then if `v` has a structural annotation `S`, set `Γ[v]` to `S`. Add any unbound shape variables in `S` to `Σ`. If `v` does not have a structural annotation, set `Γ[v]` to `ObjectStructInfo()`.
    2. In the general `VarBinding(v, e)`:
        1. If `e` is a function literal, then recursion is permitted. In this case, `v` must have a structural annotation `Sv`. Derive the structural information for `e` as follows: Set `Γ[v]` to `Sv`, apply the normal rule for function literals (given below) to `e` to derive structural information `Se`, and finally remove `v` from `Γ`. Raise an error if `Se` and `Sv` are not compatible (via `check_compatibility`).
        2. Otherwise, derive the structural information of `e` and call it `Se`.
        3. If `v` has a structural annotation `Sv`, then apply `check_compatibility` to `Sv` and `Se`. If they are compatible, then set `Γ[v]` to `Sv` (respecting the user's intent in giving an annotation). Give a warning if `Sv` is more specific than `Se`. If are not compatible, then raise an error.
        4. If `v` does not have a structural annotation, then set `Γ[v]` to `Se`.
    3. For `MatchCast(v, value, S)`:
        1. Derive the structural information of `value` and call it `Sv`.
        2. Add any new shape variables in `S` to `Σ`.
        3. If `S <: Sv` and `Sv <: S` both do not hold, give a warning, as this indicates a cast that will _always_ fail at run time. (Conversely, if `Sv <: S`, then the cast will always succeed.)
        4. If `v` is given and it has a structural annotation `S'`, then give an error if `S <: S'` does not hold. If they are compatible, then set `Γ[v]` to `S'` (respecting the user's intent in giving an annotation). (TODO: It doesn't seem very sensible to have a dynamic cast and give a different annotation, perhaps we should simply not permit doing that.)
        5. If `v` is given and it does not have a structural annotation, then set `Γ[v]` to `S`.
12. For `TupleGetItem(tuple_value, i)`:
    1. Derive the structural information for `tuple_value` and call it `St`. 
    2. Raise an error if `St` is not `TupleStructInfo`. 
    3. If `St` is `TupleStructInfo(fields)`, then raise an error if `fields` value has less than `i + 1` members.
    4. Use `fields[i]` (zero-based) as the structural information for the `TupleGetItem`.
13. For an `ExternFunc` node, the resulting structural information is `FuncStructInfo(params=None, ret=ObjectStructInfo(), derive_func=default_derive)`.
14. For `Call(op, [arg1, arg2, ..., argn], type_args=[aT1, aT2, ..., aTn])`:
    1. For a call to an `Op`:
       1. We use the manually defined `FInferStructInfo` macro if it has been defined for `op` and `ObjectStructInfo()` as the resulting `StructInfo` if it has not. `FInferStructInfo` is a function that takes in the call node and returns the structural information of the result.
       2. If the current function has `is_pure` set to `True` and the current function does not have `relax.force_pure` mapped to `True` in its `attrs` field _or_ if the current scope is inside a `DataflowBlock`, then consider it a type error if `op` does not have `True` as the value for its `FPurity` attribute.
    2. Otherwise, derive the structural information for `op` and call it `Sf`. Next derive the structural information for the args and call it `S1`, `S2`, ..., and `Sn`. 
        1. Give an error if `Sf` is not `FuncStructInfo`.
        2. If the `derive_func` field of `Sf` is defined:
            1. If the current function has `is_pure` set to `True` and the current function does not have `relax.force_pure` mapped to `True` in its `attrs` field _or_ if the current scope is inside a `DataflowBlock`, then give a type error: External functions are assumed to be impure by default (the `call_pure_packed` operator can be used to indicate to the compiler that an external function is, in fact, pure).
            2. Apply the `derive_func` macro to the call node to derive the structural information for the call node, ignoring the `ret` field of `Sf`. Additionally, 
        3. If the current function has `is_pure` set to `True` and the current function does not have `relax.force_pure` mapped to `True` in its `attrs` field _or_ if the current scope is inside a `DataflowBlock`, then consider it a type error if `Sf`'s `purity` field is not `True`.
        4. Otherwise, `params` must be defined. Give an error if the length of `params` does not match the number of call arguments. Let the members of params be `P1`, `P2`, ..., `Pn`.
        5. Next, attempt to perform [beta-reduction](https://en.wikipedia.org/wiki/Lambda_calculus#%CE%B2-reduction) by matching unbound shape variables in `params` with the `Si`. Namely, get a shape var mapping `m` by applying `get_shape_var_mapping(params[i], Si)` for all `i` and taking the union of all resulting mappings. For each shape variable `v` that occurs in `Sf`, replace it with `m[v]` if `v` is in `m`.
        6. After the substitutions, give an error if `Pi <: Si` does not hold for some `i` (give a warning if it _possibly_ holds).
        7. Use `erase_to_well_defined(Sf.ret, Γ, Σ)` as the resulting structural information.
15. For `Function(params=[v1, v2, ..., vn], body, ret_struct_info, is_pure, attrs)`:
    1. Let `S1`, `S2`, ..., `Sn` be the structural information of the parameters. If `vi` has a structural annotation, then use that annotation for `Si`; if not, use `ObjectStructInfo()`. Let `Sr` be `ret_struct_info` if it is defined and `ObjectStructInfo()` if not.
    2. If the function is bound to a `GlobalVar` `gv`, set `Γ[gv]` to `FuncStructInfo(params=[S1, S2, ..., Sn], ret=Sr, purity=is_pure)`. Still check the structural information in `body` per the below steps, however.
    3. For each of the `vi`, set `Γ[vi]` to `Si`. Additionally, add all new shape variables introduced in the `Si` to `Σ`.
    4. Derive the structural information for `body`, calling it `Sb`.
    5. Give an error if `Sb` is incompatible with `Sr` via `check_compatibility` (warn if only possibly compatible).
    6. If `ret_struct_info` is defined, use `FuncStructInfo(params=[S1, S2, ..., Sn], ret_struct_info, purity=is_pure)` as the structural information for the function. If `ret_struct_info` is not defined, use `FuncStructInfo(params=[S1, S2, ..., Sn], erase_to_well_defined(Sb, Γ, Σ), purity=is_pure)`.
    7. Remove all variables added to `Γ` and `Σ` during the above steps of the derivation.
16. For `PrimFunc(params, body, ret_type, buffer_map, attrs)` at the module level, which is bound to a `GlobalVar`:
    1. Suppose there are `n` members of `params`. For the `i`th member of `params` (let us call it `v`), let `si` be a corresponding `StructInfo` defined as follows:
        1. If `v` is not in `buffer_map`, then `si` is `PrimType(d)`, where `d` is the `dtype` field of `v`.
        2. If `v` is in `buffer_map`, then let `b` be `buffer_map[v]`. Then, `si` is `TensorStructInfo(dtype=d, shape=ShapeExpr(s), ndim=len(s), vdevice=undefined)`, where `d` is the `dtype` field of `b`, `s` is the `shape` field of `b`.
    2. The `StructInfo` for the `PrimFunc` (namely, for the `GlobalVar` to which the `PrimFunc` is bound) is `FuncStructInfo([s0, s1, ..., sn-1], TupleStructInfo([]), purity=False)`. (`PrimFunc`s work by mutating their arguments, so direct calls to `PrimFunc`s are treated as impure; in order to call a `PrimFunc` from within a `DataflowBlock`, use `call_tir`, which allocates fresh tensors for the outputs.)

### Propagating Virtual Device Information

If the `IRModule` contains `VDevice`s in its global information map, then we additionally propagate virtual device information to `TensorStructInfo` after deriving the `StructInfo` by the above rules. If no `VDevice`s are given in the global information map, then this step is omitted. (Implementation note: This is implemented in the pass `RealizeVDevice`.) Note that this propagation can only succeed if at least some `VDevice`s are manually provided, either through `StructInfo` annotations or calls to related operators like `to_vdevice`.

We use the following auxiliary procedure, in pseudocode, to set the `vdevice` field in `StructInfo`:

```python
def update_struct_info(S: StructInfo, v: VDevice) -> StructInfo:
    if S is TensorStructInfo:
        if S.vdevice is defined and S.vdevice != v:
            # this is a compile-time inconsistency
            raise error
        return TensorStructInfo(ndim=S.ndim, shape=S.shape, dtype=S.dtype, vdevice=v)
    «if S is TupleStructInfo:
        return TupleStructInfo(fields=[update_struct_info(s, v) for s in S.fields])»
    «if S is FuncStructInfo:
        if S has a defined derive_func:
            return S
        return FuncStructInfo(params=S.params, ret=update_struct_info(S.ret, v), purity=S.purity)»
    return S
```

For each Relax function in the `IRModule`, we will update the `VDevice`s in the "backward" direction by proceeding recursively:
1. For a `Function` node with return `StructInfo` `finfo` and body `body`, suppose its `StructInfo` is `finfo` (which must be `FuncStructInfo`). If the return `StructInfo` (`finfo->ret`) is `TensorStructInfo` with a defined `vdevice` field, then set the `StructInfo` of `body` to `update_struct_info(finfo, v)`. Visit `body` recursively.
2. For a `Call` node with callee `op` (with `StructInfo` `finfo`) and arguments `args` (for which the `i`th member has `StructInfo` `Si`), suppose its `StructInfo` is `S`. If `S` is a `TensorStructInfo` with a defined `vdevice` `v`, set the `StructInfo` of `op` to `update_struct_info(finfo, v)` and for the `i`th member of `args`, set the `StructInfo` to `update_struct_info(Si, v)`. Visit `op` and each member of `args` recursively.
3. For `SeqExpr(blocks=blocks, body=body)`, suppose the `StructInfo` of the entire node is `S`. If `S` is a `TensorStructInfo` with a defined `vdevice` `v`, set the `StructInfo` of `body` to `S`. Recuse down each binding block in `blocks` and each binding in each binding block.
    1. For each `VarBinding(var=var, value=value)`, let the `StructInfo` of `var` be `Svar` and the `StructInfo` of `value` be `Svalue`. If `Svar` is `TensorStructInfo` with a defined `vdevice` `v`, update the `StructInfo` of `value` to `update_struct_info(Svalue, v)`. If `Svalue` is `TensorStructInfo` with a defined `vdevice` `v`, update the `StructInfo` of `var` to `update_struct_info(Svar, v)`. Recurse down `value`.
    2. For each `MatchCast(var=var, value=value, struct_info=S)`, let the `StructInfo` of `var` be `Svar` and the `StructInfo` of `value` be `Svalue`. If `Svar` is `TensorStructInfo` with a defined `vdevice` `v`, update `S` to `update_struct_info(S, v)` and update the `StructInfo` of `value` to `update_struct_info(Svalue, v)`. If `S` is `TensorStructInfo` with a defined `vdevice` `v`, update the `StructInfo` of `var` to `update_struct_info(Svar, v)`. Recurse down `value`.
    3. Finally, recurse down `body`.
4. For all other expressions, recurse down to all child `Expr` nodes, making any changes specified in the above steps.

The type-checking procedure (the derivation rules) will have to be run again to propagate the `VDevice`s in the "forward" direction and ensure consistency after the `VDevice`s are filled in.

«After the second run of type-checking, consider it a compilation error if there are any `Call` nodes to `Op`s or `PrimFunc`s remaining where an argument has an undefined `vdevice`.»

### Note on Proving Shapes Equivalent and Eliminating Dynamic Checks

There can be some complexity involved in checking whether two shapes match during shape inference. A very simple, conservative method for determining equality is simply using alpha-equivalence: If the two shapes have the same structure, then they are equivalent. However, this method is conservative and can overlook numerical properties in `PrimExpr`s. We leave it up to compiler implementations as to whether to use more advanced methods for proving equivalence, such as attempting to use algebraic rewrite rules. (As a consequence, portability requires inserting dynamic checks wherever there needs to be a comparison of shapes.)

Note that optimizations like function inlining or constant folding could allow for simplifying many shape annotations and expressions and make it possible to conclude at compile time that shapes in more cases are equivalent. In general, developing compiler infrastructure for partial evaluation and reasoning about common situations with shape annotations may eliminate many dynamic checks.

Applying some kind of normalization or algebraic simplifications to `PrimExpr`s used in structural information and `MatchCast` bindings can also make it easier to conclude that certain dynamic checks may not be necessary by increasing the likelihood that more derive structural information could be made syntactically identical to the structural annotations. It would also be possible to generate compile-time warnings if analysis reveals that two shapes may not match (either using rewrite rules or by trying random values for shape variables and checking).

Since most dynamic structure checks are done for safety, it may be feasible to introduce a compilation mode that eliminates almost all dynamic structure checks. Some structure checks may not be possible to eliminate, since `ShapeExpr`s can use shape variables introduced in `MatchCast` brindings, so this would require some liveness analysis.

## Possible Extension: Indicating Unknown Dimensions

A further case that may be of interest might be using an explicit wildcard dimension (e.g., using `tir::Any`) to allow for dimensions to be specified as "unknown" in function return shapes. As described at present, the only way for a function to specify a partly unknown return shape is to make the entire return shape unknown (`RuntimeDepShape`), which loses partial shape information.

This addition would entail some, as `FInferStructInfo` and `derive_func` macros would have to deal with potential `tir::Any` nodes. However, the advantage of implementing it would be increasing the amount of shape information present at compile time and hence that could be used by lower levels of the compiler stack. The present defaults of using more general `StructInfo` means that either of these changes could be pursued in the future without breaking existing code, since these would generally have to be paired with explicit `MatchCast` dynamic checks, which will still work even if we add rules to automatically infer the shapes in those cases.

## Traditional Types

For comparison with Relay, it may be useful to simplify `StructInfo` into more traditional types that do not contain any expressions (such as in `TensorStructInfo` and `ShapeStructInfo`). We can define Relax types as follows:

```
Type ::=
    DynTensorType(ndim: int, dtype: DataType)
  | ShapeType(ndim: int)
  | PrimType(dtype: DataType)
  | TupleType(fields: [Type])
  | PackedFuncType()
  | FuncType(arg_types: [Type], ret_type: Type)
  | ObjectType()
```

We can "erase" `StructInfo` into types by the following procedure (in psuedocode):
```python
def erase_struct_info(si: StructInfo) -> Type:
    if si is TensorStructInfo:
        return DynTensorType(ndim=si.ndim, dtype=si.dtype)
    if si is ShapeStructInfo:
        return ShapeType(ndim=si.ndim)
    if si is PrimStructInfo:
        return PrimType(dtype=si.dtype)
    if si is TupleStructInfo:
        return TupleType(fields=[erase_struct_info(field) for field in si.fields])
    if si is FuncStructInfo:
        # this should be the case only for packed funcs
        if si.params is not specified:
            return PackedFuncType()
        return FuncType(
            arg_types=[erase_struct_info(arg_type) for arg_type in si.params],
            ret_type=erase_struct_info(si.ret))
    # only remaining case is ObjectStructInfo
    return ObjectType()
```

# Detailed Semantics

## Program Entry Point

In the `IRModule`, every mapping of a `GlobalVar` to a `Function` node or a TIR `PrimFunc` should be processed first and added to the global scope. «Global functions that have a `global_symbol` attribute should be externally linked, meaning that they can be invoked as program entry points; those that do not have a `global_symbol` attribute can be called only from within the global functions in the `IRModule`.»

The rules for evaluating `Function` nodes into closures are given below. TIR `PrimFunc`s evaluate into objects that are opaque to Relax, but are also assigned `FuncStructInfo` and can be called like closures. None of the values in global scope is mutable. Execution of a Relax function in an IR module thus begins by evaluating all globally visible functions into a form in which they can be accessed.

## Destination Devices

«If the `global_info` table in the `IRModule` contains any `VDevice`s, then execution will be distributed across the devices listed. The `to_vdevice` operator is responsible for transferring data (tensors) from one device to another and operators involving these tensors will be implemented on the appropriate device. If no `VDevice`s are listed in the `global_info`, then all (tensor) computations will take place on a single "target" device specified in compilation (if the target is a GPU, then some computations may still take place on a CPU host, but only those concerning metadata or data structures that are not tensors).»

## Evaluating Expressions

For each expression, we define how it affects the program's visible state and the order in which they are evaluated. Below, all evaluation results are passed by reference (and hence possibly alias) unless it is explicitly specified that they allocate new values.

1. The node `Constant(value)` creates a new tensor whose contents are `value`.
2. A variable (whether `Var`, `DataflowVar` , or `GlobalVar`) evaluates to the stored value for that variable in the current scope.
3. The node `Tuple([e1, e2, ..., en])` evaluates `e1` (yielding value `v1`), then `e2` (yielding value `v2`), …, and finally `en` (yielding value `vn`) in that order and returns a tuple value containing `v1`, `v2`, …, and `vn` in that order.
4. The node `PrimType(prim_expr)` evaluates the `PrimExpr` `prim_expr` first, obtaining a resulting `pv`. It then creates an immutable `PrimValue` containing `pv`.
5. The node `StringImm(s)` creates an immutable string container whose contents is `s`. It does not necessarily have to be a _new_ string container if, for example, string interning is implemented.
6. The node `DataTypeImm(dt)` creates a new immutable datatype representation.
7. The node `TupleGetItem(t, i)` is evaluated by first evaluating `t` (which, per `StructInfo` checking, must evaluate to a tuple) and then returning the `i`th field of the result.
8. The node `ShapeExpr([p1, p2, ..., pn])` evaluates the `PrimExpr`s `p1` (yielding dimension value `v1`), `p2` (yielding dimension value `v2`), …, and finally `pn` (yielding dimension value `vn`) in that order, using the current shape context, and returns a shape value whose dimensions are `v1`, `v2`, …, `vn`, in that order.
9. The node `Function([v1, v2, ..., vn], body)` returns a new closure containing the function definition itself and a mapping of any free Relax variables or shape variables in `body` to the values they hold in the current scope when the `Function` node is encountered. If the function is the RHS of a local binding, the bound variable should also be included in the closure's binding map and should be mapped to the closure itself (to allow for recursive calls). Closure capturing is done *by reference*; no values will be copied and references to captured values will alias their values in the outer scope. `DataflowVar`s are not captured by closures.
10. The node `If(cond, true_branch, false_branch)` is evaluated as follows:
    1. First `cond` is evaluated. Let the result be `r` (per `StructInfo` checking, it must be a `Bool` scalar).
    2. If `r` is true, evaluate the `true_branch` and return its result.
    3. If `r` is false, evaluate the `false_branch` and return its result.
11. The node `ExternFunc(global_symbol)` is evaluated by looking up the global symbol name and returning the `PackedFunc` if it exists (it is an error if it does not). Note that if a TIR `PrimFunc` in the `IRModule` has a global symbol attribute registered, it can be called as an `ExternFunc` using that global symbol as well.
12. The node `Call(op, [arg1, arg2, ..., argn])` is evaluated as follows:
    1. If `op` is an `Op` node, then evaluate `arg1`, `arg2`, …, `argn` in that order and call the results `a1`, `a2`, …, `an`. If there are `VDevice`s defined, then (per the `StructInfo` rules for propagating `VDevice` information), all tensor arguments must have a `VDevice` specified in their `StructInfo`; the operator must be implemented on the `VDevice` specified. In all other respects, it is up to the compiler implementation to decide how operators should be implemented (some may have an associated `PackedFunc` and others may be built into language runtime). The operator may mutate its arguments. It is also up to the operator implementation as to whether the result is newly allocated or aliases another value. «(TODO: Once we have operators for logical and AND and OR, we should also define short-circuiting semantics for those.)»
    2. Otherwise, first evaluate `op` (it must evaluate to a closure or `PackedFunc`). Next, we evaluate  `arg1`, `arg2`, …, `argn` in that order and call the results `a1`, `a2`, …, `an`. 
        1. If `op` evaluated to a closure, push a new scope onto the stack where arguments `v1`, `v2`, …, `vn` in the closure are bound to `a1`, `a2`, …, and `an`, respectively, and all variables saved in the closure are added to the scope. Evaluate the closure body in this new scope; this will be the return value of the call. Pop the scope before returning the value. (Note that the checking of the structural information of the argument result values and the body values should be done as described in the previous section.)
        2. If `op` evaluated to a `PackedFunc`, simply invoke it. `PackedFunc`s may have arbitrary side effect and are responsible for whether the result is a newly allocated value or aliases another value.
        3. Similarly, if `op` evaluates to a `PrimFunc` representation, the `PrimFunc` is directly called with its arguments (it likely mutates one or more of them as a result).
13. For the node `SeqExpr(blocks, body)`, we evaluate as follows:
    1. Push a new scope onto the stack.
    2. Iterate through the `BindingBlock`s in `blocks` in order. We will call the current one `block`. For each binding in `Block`:
        1. If the binding is `MatchCast(var, value, struct_info)`, perform the structure matching and shape variable updates as described in the structural information section. If `var` is provided, `var` will be bound to `value` in the current scope; this assignment is aliasing and no new value is allocated. If `var` is not provided, then the structural check is performed and shape variables are updated, but no new binding is introduced.
        2. If the binding is `VarBinding(var, value)`, then evaluate `value` and bind `var` to that value in the current scope; this assignment is aliasing and no new value is allocated.
        3. If `block` is a `DataflowBlock`, remove all `DataflowVar`s bound in the block from the current scope before proceeding to the next block.
    3. After iterating through the binding blocks, evaluate `body` in the current scope. That will be the return value of the `SeqExpr`.
    4. Pop the scope, removing any `Var` or shape variable bindings introduced in the `SeqExpr`.

### Optimizations

Optimizations are allowed to reorder and modify the operations of a program in any way so long as they do not change the value returned by evaluating the program or any visible behavior of the program. For the purposes of compilation, visible behaviors consist of side effects like mutating values in the program or external effects like I/O (printing to the console, creating files, etc.) and the order and number of times in which they happen.

«Within `DataflowBlock`s, it is permitted for the compiler to remove or reorder `MatchCast` operations even though this can affect the "visible behavior" of the program (since they can exit with an error). It is also permitted for the compiler to optimize away potential non-termination within `DataflowBlock`s: For example, if some pure function `f` has an integer return type and does not terminate, it is permissible to optimize `f() - f()` to 0 within a `DataflowBlock`. In general, the compiler is permitted to make programs "more defined" (terminating when the original did not terminate, not raising an error when the original raised an error) within a `DataflowBlock`, but never "less defined" (giving an error when the original did not give an error, not terminating when the original did not terminate). Outside of `DataflowBlock`s, error messages and potential non-termination must be preserved faithfully.»

For immutable containers like those for the results of `Tuple`, `ShapeExpr`, `PrimValue`, `StringImm`, and `DataTypeImm`, it is not required for the results of evaluating these expressions to be _new_ containers—it is permitted for the compiler to reuse existing objects provided that the values contained within are identical. This optimization is called [interning](https://en.wikipedia.org/wiki/String_interning). However, for operations that return new mutable values (in particular, operations that return tensor values), those _must_ be newly allocated, since reusing values can affect the behavior under aliasing.

The specification makes no guarantees about certain memory-related properties and hence also does not consider them to be "visible behaviors":

- Whether an allocation happens at a given point. Compiler implementations are permitted to reuse already-allocated memory if it would not interfere with visible state in any other way, per the aliasing rules (`PackedFunc`s or operators may mutate values that are passed to them and those mutations should be visible as per aliasing in this specification). Copying values or sharing representations (e.g., interning constants) between values may be done only if they will not affect any other visible behaviors, dependent on the aliasing behavior.
- It is entirely the domain of compiler implementations to make guarantees (or not) as to whether memory allocations will succeed.
- `PackedFunc`s or operators can, in principle, access information about the machine's state and make changes to allocation policies or the state that affect how memory allocations are performed. The specification makes no guarantees in such an event.

These semantic rules assume a single thread of evaluation on a single host machine. At this time, it is unspecified as to how Relax programs should behave if split over distinct threads or across multiple machines.

### Notable Operators

The above evaluation rules are general, but leave much room for implementations of operators to specify custom semantics. Certain operators are used to perform common operations and will be discussed here as well.

- `call_tir(prim_func, args, packed_ints, sinfo_args=[aS1, aS2, ..., aSk])`: 
    - `prim_func` must be a `GlobalVar` that denotes a `PrimFunc` in the current `IRModule` (we will call it `f`). 
    - `args` must be an expression that evaluates to a tuple of tensor values (where each member of a tuple will be a tensor argument to the `PrimFunc`). Let us call the members of the tuple `arg1`, `arg2`, ..., `argn`.
    - `packed_ints` is an optional argument. If present, it must be a shape value (with `ShapeStructInfo`). If present, we will call the dimensions of the value`shape1`, `shape2`, ..., `shapem` for convenience.
    - The `StructInfo` arguments `aS1` through `aSk` give the `StructInfo` of the results of calling the `PrimFunc`. 
    - All the `aSi` must be `TensorStructInfo` with a `shape` field consisting of a `ShapeExpr` (possibly containing shape variables) and a non-`Void` `dtype`, denoting the shape of the resulting tensors. 
    - If there is exactly one member of `sinfo_args`, then the operation returns a single tensor with that shape; if there are multiple or zero members of `sinfo_args`, then the result will have the `StructInfo` `TupleStructInfo(fields=[aS1, as2, ..., aSk])`.
    - Based on the `aSi`, the resulting tensors `r1`, `r2`, ..., `rk` will be allocated according to the sizes given in their `shape` fields. 
    - `f` will be called in destination-passing style, like so: `f(arg1, arg2, ..., argn, shape1, shape2, ..., shapem, r1, r2, ..., rk)`, omitting the `shapei` if `packed_ints` is not given. `f` is expected to mutate *only* the `ri` to give the output of the function, hence `call_tir` is considered pure.
    - «If the shape or data type of the actual result do not correspond to the `aSi`, an error is issued.» 
    - After the call, the `ri` will be returned (returning `r1` directly if there is only a single result, otherwise returning `Tuple(fields=[r1, r2, ..., rk])`).
- `call_tir_inplace(prim_func, args, inplace_indices, packed_ints, sinfo_args=[aS1, aS2, ..., aSk])`: Behaves similarly to `call_tir`, except the computation will mutate some members of `args` instead of allocating new tensors for all outputs. For each intended output, there must be a corresponding index given in `inplace_indices`: if the index is -1, then that output will be freshly allocated and the `PrimFunc` will take an "output argument" in destination-passing style corresponding to that output; otherwise, the `PrimFunc` will mutate the member of `args` with that index in-place. `prim_func` must be implemented in such a way as to mutate the appropriate arguments directly instead of taking output arguments in destination-passing style.
- `call_dps_packed(packed_func, args, sinfo_args=[aS1])`:
    - `packed_func` must evaluate to a `PackedFunc` object.
    - `args` must be a tuple; we will call its elements `arg1`, `arg2`, ..., `argn`.
    - The `StructInfo` argument `aS1` may be either a single `TensorStructInfo` (whose `shape` field _must_ be a `ShapeExpr`), which we will call `ts1`, or a `TupleStructInfo` whose fields are all `TensorStructInfo` (whose `shape` fields _must_ be `ShapeExpr`s), which we will call `ts1`, `ts2`, ..., `tsm`.
    - Let `r1`, `r2`, ..., `rm` be newly allocated tensors whose shape match the `StructInfo` args `ts1`, `ts2`, ..., `tsm`, respectively.
    - Evaluate `f(arg1, arg2, ..., argn, r1, r2, ..., rm)`.
    - «If the shape or data type of the actual result do not correspond to the `tsi`, an error is issued.»
    - Return `r1` if `aS1` is a single `TensorStructInfo`; otherwise, return `Tuple(fields=[r1, r2, ..., rm])`.
    - Note that it is assumed that `packed_func` will be pure, so `call_dps_packed` is treated as a pure operator (its `FPurity` is set to `True`).
- `call_pure_packed(func, args, sinfo_args)`: 
    - `func` must evaluate to a `PackedFunc` object.
    - `args` must be a tuple.
    - `sinfo_args` must be a non-empty list of `StructInfo`.
    - The returned value will have the semantics of `Call(func, args, sinfo_args=sinfo_args)`. However, this call will be assumed to be pure (`call_pure_packed`'s `FPurity` is set to `True`), thus allowing the call to appear inside a `DataflowBlock` or a function whose `is_pure` is set to `True`.
    - Note: This operator is intended to be be used for cases where  the user knows that calling the packed function will _in reality_ not cause any side effects. If it is used for a call that _does_ result in side effects, then the compiler may end up removing, reordering, or repeating that call; the specification makes no guarantees about the side effects in the callee in that case.
- `call_inplace_packed(func, args, inplace_indices, sinfo_args)`: Behaves identically to `call_pure_packed`, but `inplace_indices` denote that certain arguments (those with the corresponding indices) are mutated in-place. This is used to indicate intent, so that the compiler can verified that the arguments or any alias of those arguments will not be used after being mutated (this is what allows the operator to be considered pure and used within `DataflowBlock`s even though the `PackedFunc` might have side effects).
- `shape_of(t)`: Given a tensor argument `t`, it returns its shape. The return value is a shape object.
- `null_value()`: Returns a null object (treated as `ObjectStructInfo`). This is used for indicating to operators that an optional argument has been omitted.
- `hint_on_device(data, device)`: This operator acts as a "hint" to the compiler that `data` should be located on `device`. `device` is not a `VDevice` but is rather a device id (the `dev_id` field on a `VDevice`). This operator is de-sugared into calls to `to_vdevice` if `data` does not have a specified `vdevice` or it differs from `device` (if `data` already matches `device`, then the call is removed entirely).
- `to_vdevice(data, vdevice)`: Move `data` to the `VDevice` corresponding to `vdevice`. If `data` is a tensor, the tensor is copied over to `vdevice`. If `data` is a tuple, all members of the tuple (proceeding recursively for any members that are in turn tuples) are copied over to `vdevice`.