# Informal Relax Language Specification

Note: Text in «double chevrons» indicates features not present in the current prototype.

In order to develop and test Relax, it is important for compiler developers to agree on what a given program in Relax means and what makes it valid so that test cases can be evaluated independently of any particular Relax implementation. This document is intended to describe Relax's grammar constructs (its [abstract syntax tree](https://en.wikipedia.org/wiki/Abstract_syntax_tree), or AST), the semantics of its grammar (what the different constructs mean), Relax's type system and type-checking rules (what makes a Relax program valid), and its rules for reasoning about tensor shapes in detailed though still informal terms. If necessary, we may encode these rules more formally to allow for more automated analysis.

Though this document will use the TVMScript front end for some examples, specifying the mapping from Python's AST to Relax's AST will be deferred until the parser becomes more stable.

# Table of Contents

1. [Overview](#overview)
2. [Top-Level Program Organization](#top-level-program-organization-irmodule)
3. [Values in Relax](#values-in-relax)
4. [Variable Scoping](#variable-scoping)
5. [Well-Formedness Criteria](#well-formedness-criteria)
6. [Types in Relax](#types-in-relax)
7. [Shapes in Relax](#shapes-in-relax)
8. [Semantics](#detailed-semantics)

# Overview

This section will outline the grammar of Relax and give very brief descriptions of the different components, including the semantics, type system, and shape system. The rest of this document will provide more detailed descriptions of these facets of the language, including the validity conditions that the type system and shape system uphold.

## Differences from Relay

Per the [original workshop paper](https://arxiv.org/abs/1810.00952) and the [later report](https://arxiv.org/abs/1904.08368), Relay was designed to be a high-level functional language for expressing deep learning models at a high level. While Relay is not entirely pure (the `Ref` type is modeled after reference types in SML and similar functional languages), the assumption in Relay is that tensor operators are generally pure, meaning that they do not change the program state other than by producing new values. Additionally, Relay's type system also requires operators to have type relations that infer static tensor types or conclude that a dimension is unknown at compile time (`Any`). The need to register type relations and ensure operators' purity makes it difficult to add new operators to Relay and particularly difficult to call directly into TIR or external libraries, which are often not pure; any such extension requires adding new operators and abstracting over any impurity.

While Relax aims to be as general and expressive as Relay, Relax is intended to make it much easier to interoperate with external libraries and especially with TIR. In particular, Relax includes a mechanism for calling arbitrary TVM `PackedFunc`s (which can call external libraries) and special support for TIR. The language accordingly does not assume that such operations are pure, though this does require reasoning about aliasing and similar issues. Additionally, tensor shapes are no longer handled during type checking; each expression has an associated shape _computation_ associated with it, in addition to a type. These shape computations support static reasoning about shapes in many cases, but also facilitate a fallback to dynamic checking when that is not possible. This approach to shapes allows for richer shape constraints to be checked at run time (such as with _symbolic_ shapes, where some dimensions are variables) and allows for more quickly integrating calls into TIR or external libraries into Relax code by obviating the need for type relations.

## Grammar

Below is a diagram of the various AST constructs in Relax, including types. In code, these are defined on the C++ side in `include/tvm/relax/{expr.h, type.h}` and in Python in `python/tvm/relax/{expr.py, ty.py}`. This diagram will give the names of the AST nodes and the types and names of their members. The semantics will describe what computation each construct represents; an AST is simply data. A Relax program consists of an `IRModule` with global variables bound to Relax functions that implement the computations of interest.

(On the notation: `[x]` means "a list of `x`," `x?` means "optionally `x`," `{x: y}` means "a map of `x` to `y`," `x | y` means "`x` or `y`," and `#` is used for comments.)

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

Type ::=   DynTensorType(ndim: int, dtype: DataType)
         | ShapeType()
         | ObjectType()
         | TupleType(fields: [Type])
         | FuncType(arg_types: [Type], ret_type: Type, «pure: bool»)

DataType ::=
           Int(bitwidth: int)
	 | Float(bitwidth: int)
	 | Bool()
	 | Void()

# expressions
Expr ::=   Constant(data: NDArray)
           # scoped to functions or SeqExprs
         | Var(name_hint: string)
           # scoped to DataflowBlocks
         | DataflowVar(name_hint: string)
         | GlobalVar(name_hint: string)
         | Tuple(fields: [Expr])
         | SeqExpr(blocks: [BindingBlock], body: Expr)
         | Function(params: [Var], body: Expr, ret_type: Type?, attrs: Attrs?)
         | If(cond: Expr, true_branch: Expr, false_branch: Expr)
         | ExternFunc(global_symbol: string)
         | Call(op: Expr, args: [Expr], type_args: [Type], attrs: Attrs?)
         | ShapeExpr(values: [PrimExpr])
         | TupleGetItem(tuple_value: Expr, index: int)
         | Op(op_name: string)
         | RuntimeDepShape()

# binding blocks (analogous to sequence of statements)
BindingBlock ::= 
           BindingBlock(bindings: [Binding])
         | DataflowBlock(bindings: [Binding])

# bindings (analogous to statements)
Binding ::= 
           VarBinding(var: Var|DataflowVar, value: Expr)
         | MatchShape(var: (Var|DataflowVar)?, pattern: [PrimExpr], value: Expr)

# Relax programs are IRModules. Modules may bind global variables either to
# Relax functions or TIR PrimFuncs (specified separately).
# The Relax compiler may analyze and modify the TIR PrimFUncs as well.
Program ::= IRModule(funcs: {GlobalVar: Function|PrimFunc})
```

## Expression Survey

This specification provides a more detailed description of what each expression and type represents and what conditions make them valid. To motivate and provide more context for the full specification later in this document, this section will briefly summarize the purpose of each node.

1. `Constant` nodes construct tensor constants (n-dimensional arrays of scalars).
2. `Tuple` nodes construct a tuple (fixed-size ordered grouping) of Relax values.
3. `Var`, `DataflowVar`, and `GlobalVar` nodes are all variables, referring to named stored values of different kinds. Variables in Relax must be bound exactly once. `GlobalVar`s are bound in the `IRModule` itself and refer to Relax functions or TIR `PrimFunc`s. `Var` nodes are bound either within functions, where they represent function parameters, or in `VarBinding` or `MatchShape` nodes in `BindingBlock`s, as we will discuss below. `DataflowVar`s are similar to `Var`s and can be bound only within `DataflowBlock`s.
4. `PrimExpr`s are used to represent dimensions of shapes in `ShapeExpr` and `MatchShape` nodes. These represent operations on integers with their own `Var` nodes (`tir::Var`), which we will refer to as "shape variables". Shape variables can only be used in other `PrimExpr`s and are scoped like `Var` nodes (`relax::Var`), which we will call "Relax variables."
5. `Call` nodes represent function calls. The callee argument (the `op`) can be an `ExternFunc` node (representing a call to a `PackedFunc`), an `Op` node (representing a call to a Relax operator), or an arbitrary expression. 
    1. For `ExternFunc` nodes, the call will look up the registered `PackedFunc` by its global symbol and will call it with the given arguments (note that a TIR `PrimFunc` can be compiled into a `PackedFunc` and called using `ExternFunc` by defining a `global_symbol` attribute in the `PrimFunc`). «The attribute "pure" can be specified on a call to an `ExternFunc` to indicate that it performs no side effects (for use inside `DataflowBlock`s).»
    2. `Op` nodes refer to built-in Relax operators, which the compiler is free to implement as is deemed appropriate. Certain operators implement important operations, like `call_tir` (allows for calling TIR `PrimFunc`s) «and `cast` (performs dynamic type conversions).»
    3. Any other expression must evaluate to a closure; the closure will then be called with the given arguments. 
    
    Calls to `ExternFunc`s and operators may perform side effects, hence it is important to reason about whether a function call is permitted inside a `DataflowBlock`.
    
6. `If` nodes represent branching control flow. First the condition expression is evaluated, and it must evaluate to a Boolean scalar. If the condition is true, the true branch is evaluated and its result is used; otherwise, the false branch is evaluated and its result is used.
7. `TupleGetItem` nodes represent tuple indexing. The `tuple_value` expression must evaluate to a tuple with at least `index + 1` items and the item with the given index will be returned.
8. `SeqExpr` describes a sequence of binding blocks followed by a return expression. The `SeqExpr` opens a new scope. Its binding blocks are evaluated in order and add new variables to the scope. Binding blocks are either ordinary `BindingBlock`s or `DataflowBlock`s and both consist of a series of bindings. `DataflowBlock`s are the only kind allowed to introduce bindings with `DataflowVar`s and it does not permit any constructs featuring control flow (`If` nodes or recursive calls) or calls to (possibly) impure functions. There are two different kinds of bindings:
    1. `VarBinding`s: The `value` expression (the right-hand side of the binding) of the binding is evaluated first and is bound to the `var` expression, which must be a new `Var` or `DataflowVar` (in a dataflow block). The newly bound variable will have that value for the remainder of the scope (`DataflowVar`s are scoped only to the `DataflowBlock` in which they appear; `Var`s are scoped to the entire `SeqExpr`).
    2. `MatchShape`s: The `value` expression is evaluated and the resulting shape is dynamically checked against the shape denoted by the `PrimExpr`s in the `pattern` field.
        1. If `value` evaluates to a tensor value, the pattern will be checked against the shape of the tensor; if it evaluates to a shape value, the pattern will be checked directly against the shape.
        2. Any shape dimension in the pattern that consists of a single new shape variable is treated as a binding: The variable is bound to the size of the corresponding dimension of the value being matched. 
        3. If the shapes do not match, an error is triggered. If there is a variable provided, the value is bound to the `var` expression (if the variable is omitted, the shape check is performed and any shape variables are updated, but no new binding is introduced). Shape variables introduced in a `SeqExpr` are similarly scoped to the `SeqExpr`.
    
    The `SeqExpr`'s `body` expression is allowed to reference any `Var`s introduced within the `SeqExpr`'s binding blocks in addition to those that were in the outer scope; the `body` expression is evaluated after the binding blocks and its value is what is returned. Any Relax variables and shape variables introduced in the `SeqExpr` are removed from scope after the expression finishes evaluating.
    
9. `ShapeExpr` nodes construct shape literals. The `PrimExpr`s within it describe how to compute each dimension; they are free to use any shape variables that are in scope.
10. `Function` nodes represent function definitions, taking in the listed parameters and evaluating the body expression in a new scope (meaning any variables defined from within the function cannot be referenced outside it). Function definitions may be nested in any other expression and they evaluate into closure values, ensuring that functions are first-class. Closures capture any variables from the outer scope that are used in their body, both Relax variables and shape variables. Note that function definitions themselves are anonymous—a function must be registered in the `IRModule` (bound to a `GlobalVar`) or appear on the right-hand side of a binding to have a name in order to be called recursively. 
    
    The function can have shape annotations on the parameters and a return shape parameter. When the function is called, the annotations on parameters checked against the argument values in similar fashion to `MatchShape` and can introduce new shape variables that are scoped to the function.
    
    «A function mapped bound to a `GlobalVar` can have a `global_symbol` attribute defined to indicate that it should be externally linked externally (be accessible outside the `IRModule`). The absence of a `global_symbol` attribute on a function definition bound to a `GlobalVar` indicates that it is "private" and hence can be called only within the `IRModule`.»
    
11. `RuntimeDepShape` nodes are used to denote that a shape is unknown at compile time and must be deduced at run time. These nodes may appear only in shape annotations and have no run-time semantics of their own.

## Purity and Dataflow Blocks

A function or operator is called "pure" if it does not have side effects, which refers to any change in program state besides returning a result. Side effects include mutating values other than those they create, aborting the program, or file I/O (including writing to the console). Purity is a useful property for compiler optimizations, since calls to pure functions can be reordered or duplicated or (if the result is unused) eliminated without changing any other program behavior. Most deep learning operators are pure, as they perform arithmetic on tensors and return a new tensor containing the result. «In Relax, we conservatively assume that any function that calls an impure function is itself impure, though the attribute `force_pure` on a function can be used as an override (e.g., if a function creates a new tensor, mutates it, and returns it, that is still pure but does not satisfy the conservative rule).»

Above, it is mentioned that `DataflowBlock`s are not allowed to contain constructs featuring control flow (`If` nodes or recursive calls to the current function) or calls to impure functions. This ensures that `DataflowBlock`s represent a directed acyclic graph of pure operations, which is similar to the graph-like abstractions of traditional deep learning frameworks. This allows many common optimizations from past frameworks to be directly adapted to `DataflowBlock`s without having to accommodate additional reasoning about more expressive features like control flow and side effects.

There is one visible side effect that Relax permits inside otherwise "pure" functions, namely exiting the program with an error. This can arise in the following cases:

- Shape matching errors (from `MatchShape` or from implicit shape checks upon calling a Relax function)
- Errors raised by otherwise pure Relax operators or `PackedFunc`s, such as in `cast` (which dynamically checks types). Since the purity of operators or `PackedFunc`s must be manually registered, this means that it is permissible to register an operator or `PackedFunc` as being pure if its only side effect is issuing an error in some cases.

Even though an abnormal program exit is a visible side effect and removing or reordering it changes the observable semantics, it would be too great a restriction to prohibit error checking inside `DataflowBlock`s. Relax does not have any notion of exception handling, so the only consequence of a failed safety check can be exiting the program. It is permissible for the compiler to reorder, duplicate, or eliminate `MatchShape`, `cast`, or otherwise pure operations that have the potential of failing, provided that doing so does not change the value returned by the program or any other visible behavior.

To indicate that an operator or `PackedFunc` that can abort with an error should *never* be reordered or removed by the compiler, it should *not* be marked as pure. However, this means that it cannot be used inside a `DataflowBlock`.

Note that in some programming languages like Koka, non-termination is also considered a side effect, since it can in some sense be "observed" by a user and affects the visible behavior of a program (e.g., if there is an infinite loop before a print statement, the print will never happen). However, since non-termination cannot be automatically detected in general and is unlikely to arise in deep learning models, we do not attempt to systematically track non-termination in Relax. In general, the Relax compiler is allowed to reorder or remove otherwise pure function calls even if they may not terminate. For example, if a pure function `f` that returns an integer scalar does not terminate, it is permissible in principle to rewrite `f() - f()` to 0.

Exiting with an error and infinitely looping are traditionally considered "[divergence](https://en.wikipedia.org/wiki/Divergence_(computer_science))" in the programming languages literature. As a general principle, Relax's compiler is permitted to turn a program that diverges into a program that does not diverge (provided that no other visible effects change) so long as it never transforms a program that does not diverge into one that diverges.

## Type System Survey

The types in Relax correspond to the broad categories of the values given above:

1. `DynTensorType` corresponds to tensor values, giving the scalar data type and the number of dimensions (rank), both of which are optional.
2. `TupleType` corresponds to tuple values, giving the type of each member of the tuple.
3. `ShapeType` corresponds to shape values.
4. `FunctionType` corresponds to function values (closures), giving the types of the parameters, the return type, «and whether the function is pure.»
5. `ObjectType` is the parent type of all Relax types and corresponds to all the values above as well as any values returned by `PackedFunc` calls that do not fit in the above categories.

The type checking rules assign types to every variable in scope and every type of expression based on the values it returns, making use of subtyping to assign more general types when a more specific one cannot be determined. «Relax is strongly typed, meaning that if a type encountered is less specific than the one expected, an error will be issued and an explicit cast (via the `cast` operator) will be required.»

## Shape System Survey

In Relax, tensor shapes are not handled in the type system; each expression instead a has an associated shape expression. In many cases, these shape computations can allow for statically concluding that two shapes are the same and thus eliminate the need for dynamic checks via `MatchShape`. However, when shapes cannot be statically concluded to be the same, it may be necessary for there to be dynamic checks. The compiler is also free to make use of shape expressions for memory planning purposes. «Relax is "strongly shaped," meaning that if the compiler cannot conclude that shapes match in certain cases, an error will be issued and an explicit `MatchShape` will be required.»

---

# Top-level Program Organization: `IRModule`

As with Relay, the top level of organization for a Relax program is an `IRModule`. An `IRModule` contains mappings of global variables to functions, both Relax functions as well as TIR functions (which can be called from Relax). The global function called `main` is usually considered the entry point to the program (meaning that execution starts by calling that function), though any function with a `global_symbol` attribute can be specified as the entry point during compilation. In the AST (see below), the names of Relax functions in the `IRModule`s are `GlobalVar` nodes.

Oftentimes, compiler passes operate only on particular functions or add new functions to the `IRModule`, but a pass can operate over the entirety of a Relax program by iterating through all the functions in an `IRModule`.

# Values in Relax

Here are the classes of values that Relax operates over, meaning that they can be assigned to variables or be the result of evaluating expressions.

- *Tensors* are n-dimensional arrays of scalar values (which can be integers of fixed bitwidths, floats of fixed bitwidths, or bools). A tensor's *shape* is a tuple of the size of each dimension; the number of dimensions is a tensor's *rank*. For example, a vector (1, 2, 3) is a rank-1 tensor of shape `(3,)`. Note that scalars are tensor values with a rank of 0, meaning that their shape is `()`.
- *Tuples* represent a fixed-size grouping of other Relax values (tensors, closures, shapes, objects, or other tuples, to an arbitrary degree of nesting). Note that an empty tuple, i.e., `()`, also called "unit" in functional programming, is commonly used as the return value for operations return no value (as may be the case in some `PackedFunc` or operator calls that have side effects).
- *Closures* are the values resulting from evaluating Relax function expressions; closures can be passed around like other values, ensuring that functions are first-class in Relax. Functions defined in Relax can capture variables from outer scopes. A [closure](https://en.wikipedia.org/wiki/Closure_(computer_programming)) consists of a function and a mapping of any variables "captured" (those are *free variables* in the function body, variables from an outer scope that are neither arguments nor defined within the function but are used in the function) to their values. Closures capture both Relax-level local variables and shape variables from outer scopes. A closure also stores a name for itself when the body contains recursive calls. «Closures additionally carry some *run-time type information* (RTTI) indicating their argument types and result type, in order to facilitate dynamic type checks (since it is not otherwise possible to introspect the function contained within a closure); the precise form of the RTTI is left up to the compiler implementation to determine so long as the `cast` operator can verify the type of a closure. Closures can be evaluated in a call node, which results in calling the function with the call's arguments and the captured values.»
- *Tensor shapes* (shape values) are tuples of integers describing a tensor shape, obtained by evaluating `ShapeExpr`s.
- Additionally, there are further  *arbitrary objects* that do not belong in the above categories. These can be returned by `PackedFunc`s and operators; additionally, we treat TIR `PrimFunc`s as opaque objects. Though Relax expressions other than `PackedFunc` and operator calls cannot use those objects, Relax should pass around these values faithfully. In the future we may add more value types in order to distinguish between different objects, but at present we treat these all as arbitrary values of type `ObjectType`.

## Representation of Values at Run Time

Because Relax supports calls to arbitrary `PackedFunc`s that can operate on a low level, it is necessary to define a convention for how values will be represented at run time. At this time, the specification does not require any specific representation and permits compiler implementations to choose their own representations, provided that each value type listed above can be recognized at run time (for dynamic type checks). This means that Relax programs that call `PackedFunc`s directly are not portable across compiler implementations: The `PackedFunc`s used must be able to operate on the run-time representations of values.

Possible specification in terms of the TVM object system:

- Tensors are represented at run time as `NDArray`s (see `include/tvm/NDArray.h`).
- Tuples are represented using TVM ADTs (algebraic data types), which are arrays of TVM objects with a tag (see `include/tvm/runtime/container/adt.h`). Tuples use a tag of 0.
- At run time, closures are represented as a `ClosureObj` (see `include/tvm/runtime/container/closure.h`); in the Relax VM these more specifically use the `VMClosureObj` (see [`https://github.com/tlc-pack/relax/blob/relax/include/tvm/runtime/relax_vm/executable.h`](https://github.com/tlc-pack/relax/blob/relax/include/tvm/runtime/relax_vm/executable.h)).
- Shape values are represented at run time as a `ShapeTuple` (see `include/tvm/runtime/container/shape_tuple.h`).
- We require objects other than the above values used by and returned by `PackedFunc` to inherit from TVM's `Object` class (defined in `include/tvm/runtime/Object.h`). Note that `PackedFunc`s are capable of using and returning all TVM POD (plain-old data) values (see `include/tvm/runtimes/packed_func.h`), which includes some representations that do not inherit from `Object`. In the future, we may define semantics for other values, but at present, these are *unsupported* in Relax and we make no guarantees about the semantics of calling `PackedFunc`s that use or return anything that does not inherit from `Object`.

# Variable Scoping

There are four relevant scopes in Relax, which determine where variables are visible and can be used:

1. Global: `GlobalVar`s can be referenced from any function in the `IRModule`, whether a Relax function or a TIR `PrimFunc`. All global functions are visible to each other and to themselves, allowing for mutual recursion.
2. Function: The parameters to a function (ordinary `Var` nodes) can be referenced from anywhere in that function. In a recursive binding (a `Binding` node where the RHS is a `Function` node or `GlobalVar` being mapped to a function at the `IRModule` level), the variable being bound is also scoped to that function, allowing for defining a recursive function.
3. `SeqExpr`: `Var` nodes defined in a `BindingBlock` in a `SeqExpr` node can be referenced in any later binding within the same `BindingBlock`, in any binding within any later `BindingBlock` in that `SeqExpr` node, or in the `SeqExpr`'s body expression. The variables defined in the `BindingBlock`s leave scope once the `SeqExpr` returns.
4. `DataflowBlock`: `DataflowVar`s introduced in a `DataflowBlock` can be referenced in any later binding within that `DataflowBlock`, but leave scope *once that `DataflowBlock` finishes executing*. Definitions in a `DataflowBlock` that are intended to leave the `DataflowBlock` should be bound to an ordinary `Var`.

Note that Relax variables must be bound _exactly_ once. A global variable is bound if it is mapped to a function in the `IRModule` and a local variable is bound if it appears as a function parameter or if it appears on the left-hand side (LHS) of a binding (`VarBinding` or `MatchShape`).

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

# Well-Formedness Criteria

Prior to type-checking and shape inference, Relax programs must conform to certain syntactic criteria to be valid.

1. `DataflowVar`s can be bound only inside `DataflowBlock`s. Additionally, a `DataflowVar` may not be used outside of the `DataflowBlock` in which it is defined.
2. A `Var` of any kind used in the program must be either a function parameter or appear on the LHS of a binding exactly once. In the binding where a `Var` is defined, the same `Var` is permitted to occur in the RHS of the binding only if the binding is defining a function (i.e., local functions are permitted to be recursive).
3. A `Var` of any kind may not appear before it is bound. Namely, if a `Var` is bound in a `BindingBlock` in a `SeqExpr`, that `Var` may not appear in bindings that precede the one where it appears on the LHS.
4. «A return shape annotation for a function is not allowed to use any shape variables that are not in scope at the function definition. That is, the only shape variables that can appear on the return shape annotation are those defined in the outer scope or those introduced in the argument shape annotations.»
5. In each function, `PrimExpr` variables (shape variables) similarly may not appear in `ShapeExpr`s or shape annotations before the shape variables are bound (either in function signatures or `MatchShape` bindings). A shape variable is bound only when it appears in a dimension by itself (for example, a dimension consisting of `x` will bind `x`; however, `2*x` is not a binding and is considered an error if `x` has not yet been bound) in a `MatchShape` node or a function argument shape annotation.
6. The following constructs are not permitted to occur inside `DataflowBlock`s, which must be side effect– and control flow–free: 
    1. Recursive calls to the current function
    2. Calls to a global function that is mutually recursive with the current function
    3. `If` nodes
    
    «Calls to Relax functions, `ExternFuncs`, or `Op`s that are not pure are also not permitted, but this must be detected during type checking.»
    
7. «For functions that contain recursive calls to themselves or mutually recursive global functions (i.e., those where function `a` calls function `b` and function `b` calls function `a`), a return type annotation is *required*. [TODO: Do we also require a return shape annotation in such cases?]»
8. `Op` nodes may appear only as the `op` argument to `Call` nodes. 
9. `ExternFunc` expressions may appear only as the `op` argument to `Call` nodes.
10. The `type_args` field is used only for type checking calls of `ExternFunc`s and `Op`s. Calls to `ExternFunc`s must have exactly one type argument, indicating the return type. Calls to `Op`s may use `type_args` as they wish. No other calls may have a non-empty `type_args`.
11. If a variable has both a type annotation and a shape annotation, the `ndim` of any `DynTensorType`s must match the number of dimensions in the corresponding shape annotation.
12. A function definition inside a `DataflowBlock` may not use `DataflowVar`s from the outer scope in its body. We do not define closure capturing for `DataflowVar`s.
13. «At least one global function in the `IRModule` must be externally linked (have a `global_symbol` attribute) in order to serve as a program entry point.»
14. «If a global function has a defined `global_symbol` attribute, the `global_symbol` name must be the same as the `GlobalVar`'s name hint.»
15. «Any `PackedFunc` or operator called in a shape annotation or `shape_` expression must be pure and be annotated as such.»
16. The node `RuntimeDepShape` may appear only in shape annotations and `shape_` expressions. It has no defined semantics at run time.

# Types in Relax

Relax presently has five types, defined in the implementation in `python/tvm/relax/ty.py` and `include/tvm/relax/type.h`:

1. `DynTensorType`, referring to tensor values (referred to in the front-end as `Tensor`). In Relax, tensor types keep track of the rank (number of dimensions) in the `ndim` field and the data type of the tensor data in the `dtype` field. Both the rank and data type are optional: Using -1 for `ndim` indicates that the tensor is of unknown rank and using `DataType::Void()` for  `dtype` indicates that it's of unknown data type.
2. `ShapeType`, referring to shape values.
3. `FuncType`, referring to functions (closures). `FuncType`s specify the types of their parameters, a return type, and whether the function is pure.
4. `TupleType`, referring to tuple values, giving the types of their fields.
5. `ObjectType`, referring to any Relax value, including values used and returned by `PackedFunc` or operator calls that do not belong in any of the above categories.

## Subtyping

Relax implements subtyping, which means that members of types can be accepted where members of their supertypes are accepted. We will denote the subtyping relationship as `T1 <: T2`, indicating that `T1` is a subtype of `T2`. For example. if `T1 <: T2` and some function expects an argument of type `T2`, then passing a member of type `T1` to that function is permitted; passing a member of type `T2` as an argument to a function that expects type `T1` for that argument is *not* permitted—the value would have to be dynamically cast to `T1` using the `cast` operator.

### Rules for Subtyping

1. Reflexivity: For all types `T`, `T <: T`.
2. Transitivity: For all types `T1`, `T2`, and `T3`, if `T1 <: T2` and `T2 <: T3`, then `T1 <: T3`.
3. For all types `T`, `T <: ObjectType`. Hence, `ObjectType` is a supertype to all Relax types (all values in Relax are members of `ObjectType`).
4. Rules for `DynTensorType`:
    1. For all fixed  `ndim`  values  `m`, where `m` ≥ 0, and `dtype`s  `d`, `DynTensorType(ndim=m, dtype=d) <: DynTensorType(ndim=m, dtype=Void)`.
    2. For all fixed `ndim`  values `m` and `dtype`s `d` that are not `Void`, `DynTensorType(ndim=m, dtype=d) <: DynTensorType(ndim=-1, dtype=d)`.
    3. Corollary: `DynTensorType(ndim=-1, dtype=Void)` is a supertype to all tensor types, since it refers to any possible tensor value.
5. Suppose we have types `T1 <: T1'`, `T2 <: T2'`, …, `Tn <: Tn'`. Then `TupleType(fields=[T1, T2, ..., Tn]) <: TupleType(fields=[T1', T2', ..., Tn'])`.
6. Rules for `FuncType`:
    1. Impure functions are supertypes to pure functions. Namely, if we have types `T1`, `T2`, …, `Tn` and `Tr`, then `FuncType(arg_types=[T1, T2, ..., Tn], ret_type=Tr, pure=True) <: FuncType(arg_types=[T1, T2, ..., Tn], ret_type=Tr, pure=False)`.
    2. Suppose we have types `T1' <: T1`, `T2' <: T2`, …, `Tn' <: Tn` and `Tr <: Tr'`. Then `FuncType(arg_types=[T1, T2, ... Tn], ret_type=Tr, pure=p) <: FuncType(arg_types=[T1', T2', ..., Tn'], ret_type=Tr', pure=p)`. Note the direction of the subtyping relationships for the argument and return types: We must be able to *call* this function with the *same* arguments and *use the returned value* wherever it is accepted—hence a function that takes more general arguments and returns a more specific return value can be used in place of the original.

These rules allow us to define the least upper bound (LUB) for any two types `T1` and `T2`, meaning that it is the most specific type `T` for which `T1 <: T` and `T2 <: T` ("most specific" meaning that if there exists some other `T'` for which `T1 <: T'` and `T2 <: T'`, then `T <: T'`). The LUB is guaranteed to exist for any two types because `Object` is a supertype to all types.

Note that the rule for obtaining the LUB of function types relies on the counterpart to the LUB, the greatest lower bound (GLB). The GLB is not guaranteed to exist for any two types in Relax, as there is no single type that is a subtype of all others.

We can give an algorithm for determining the LUB and GLB for two types, in pseudocode:

```python
def find_glb(T1 : Type, T2 : Type) -> Type?:
    if T1 == T2: # syntactic equality
        return T2
    if T1 is ObjectType:
        return T2
    if T2 is ObjectType:
        return T1
    if T1 and T2 are not both DynTensorType, not both TupleType, not both FuncType, or not both ShapeType:
        return None
    if T1 and T2 are both DynTensorType:
        ret_ndim = T1.ndim
        ret_dtype = T1.dtype
        if ret_ndim == -1:
            ret_ndim == T2.ndim
        if ret_dtype == Void:
            ret_dtype = T2.dtype
        if ret_ndim != -1 and T2.ndim != ret_ndim:
            # mismatch, so there's no common lower bound
            return None
        if ret_dtype != Void and T2.dtype != ret_dtype:
            return None
        return DynTensorType(ret_ndim, ret_dtype)
    if T1 and T2 are both TupleType:
       if they do not have the same length:
           return None
       fields = []
       for field1, field2 in zip(T1.fields, T2.fields):
           glb = find_glb(field1, field2)
           if glb is None:
              return None
           fields.append(glb)
       return TupleType(fields)
   if T1 and T2 are both FuncType:
      «if they are not both pure or both impure:»
         «return None»
      purity = T1.purity
      if they do not have the same arity:
         return None
      # mutual recursion with finding the LUB
      arg_types = [
          find_lub(arg_type1, arg_type2) 
          for arg_type1, arg_type2 in zip(T1.arg_types, T2.arg_types)
      ]
      ret_type = find_glb(T1.ret_type, T2.ret_type)
      if ret_type is None:
         return None
      return FuncType(arg_types, ret_type, purity)

def find_lub(T1 : Type, T2 : Type) -> Type:
    if T1 == T2: # syntactic equality
        return T1
    if T1 or T2 is ObjectType:
        return Object
    if T1 or T2 are not both DynTensorType, or both TupleType, or both FuncType, or both ShapeType:
        return ObjectType
    if T1 and T2 are both DynTensorType:
        res_ndim = T1.ndim
        res_dtype = T1.dtype
        if T1.ndim != T2.ndim:
            res_ndim = -1
        if T1.dtype != T2.dtype:
            res_dtype = Void
        return DynTensorType(res_ndim, res_dtype)
    if T1 and T2 are both TupleType:
        if they do not have the same length:
            return ObjectType
        return TupleType([
            find_lub(field1, field2) 
            for field1, field2 in zip(T1.fields, T2.fields)
        ])
    if T1 and T2 are both FuncType:
        «purity = (True iff they're both pure)»
        if they do not have the same arity:
            return ObjectType
        arg_types = []
        for arg_type1, arg_type2 in zip(T1.arg_types, T2.arg_types):
            # potential mutual recursion
            glb = find_glb(arg_type1, arg_type2)
            if glb is None:
                return ObjectType
            arg_types.append(glb)
        return FuncType(arg_types, find_lub(T1.ret_type, T2.ret_type), «purity»)
```

### When Type Conversions are Necessary

For two types `T1` and `T2`, if `T1 <: T2`, then a value of type `T1` can be passed anywhere a value of type `T2` is expected without any need for type conversions or dynamic checks.

*However*, if `T1 <: T2`, then passing a value of type `T2` where `T1` is expected can only be done if there has been some kind of dynamic check or conversion of that value. «Relax is *strongly* *typed*, meaning that the compiler will give an error in this situation and require an explicit conversion via the `cast` operator, which inspects the value's run-time representation and exits the program with an error message if the value is not a subtype of T1.»

If `T1` is not a subtype of `T2` and `T2` is not a subtype of `T1`, then it is always a type error to pass a value of either type where a value of the other is expected (no member of either type can be a member of the other).

## Type Checking Rules

The type checking rules for Relax are relatively simple and allow in some cases for types to be inferred without user annotations. Below, we describe how the types for each expression can be derived and when type checking should return an error.

Let us consider a typing context `Γ`, which is a map of variables to types.

1. «We type check the entire `IRModule` one function definition at a time. To handle mutual recursion, we prepopulate `Γ` with the annotated types of all global functions that are called mutually recursively. We then proceed to check the types of the global functions one at a time.»
2. Given a variable `v`, if `v` is in `Γ`, then we return `Γ[v]`. Otherwise, it is an error.
3. Given a constant expression `Constant(data)`, the type is `DynTensorType(ndim=n, dtype=d)` where `n` is the number of dimensions in `data` and `d` is its data type (recall that `data` is an `NDArray` literal).
4. Given a shape expression `ShapeExpr(dims)`, its type is `ShapeType`.
5. The type of a `RuntimeDepShape` expression is `ShapeType`.
6. Given a tuple literal `Tuple([e1, e2, ..., en])`, suppose that `e1` has the type `T1`, `e2` has the type `T2`, …, and `en` has the type `Tn`. Then the type is `TupleType([T1, T2, .., Tn])`.
7. Given a call node `Call(op=op, args=[a1, a2, ..., an], type_args=[aT])`:
    1. If `op` is a Relax `Op` node, then we look up its registered `FInferType` property. `FInferType` is a macro that takes in the `Call` node and produces a type. We return the type `op.FInferType(Call(op, [a1, ..., an], type_args=[aT]))`. The implementation of `FInferType` is free to throw errors.
    2. If `op` is `ExternFunc`, then use the sole member of `type_args` (calls to `ExternFunc`s are required to have exactly one `type_args` member) `aT` as the return type. Packed functions may be passed any combination of values and return any value; it is the responsibility of the packed function itself to do any validation.
    3. Otherwise, check the types of the subexpressions, left to right. Suppose `op` has the type `Tf`, `a1` has type `T1`, …, and `an` has type `Tn`. If `Tf` is not a function type with exactly `n` arguments, we consider it a type error and require an explicit cast. Suppose `Tf` has argument types `T1'`, `T2'`, …, `Tn'`. Consider it a type error if any of the following does not hold: `T1 <: T1'`, `T2 <: T2'`, …, or `Tn <: Tn'`. Then the return type is `Tf.ret_type`.
8. Given a conditional expression `If(cond, true_branch, false_branch)`, we first assert that the type of `cond` is `DynTensorType(ndim=0, dtype=bool)`, giving an error otherwise. Next, we recursively check the types of `true_branch` and `false_branch`; suppose they yield types `Tt` and `Tf`, respectively. The return type will be `T = LUB(Tt, Tf)`.
9. For a `TupleGetItem(t, i)` expression, suppose that the type of `t` is `T`. If `T` is a tuple type with at least `i` members, then return the `i`th type in `T`. «Give a type-checking error and require an explicit cast if `T` is not a tuple type with at least `i` members.»
10. Let us consider a sequence expression `SeqExpr(blocks = [b0, b1, ..., bn], body)`.
    1. We type check the binding blocks `b0`, `b1`, …, `bn` in order. For each block, we go through the bindings in order.
        1. «If the current block is a `DataflowBlock`, consider it an error if any binding contains a call to an expression with a function type that is not pure, a call to an `ExternFunc` that does not have the `pure` attribute, or a call to an `Op` that does not have a `pure` attribute.»
        2. For each binding `VarBinding(v : T, e)` in the current block, where `T` is the optional annotation on `v`, check the type of `e` and suppose it is `T'`. If `T` has been omitted, then add `v` to `Γ` with type `T'`. If `T` has been defined, then emit an error if `T'` is not a subtype of `T` and add `v` to `Γ` with type `T`. «If `T'` is a supertype of `T`, emit an error and require a cast.» Note that this means that annotated types can be *less specific* than the inferred type and that a user annotation forces the type system to consider the variable as having a less specific type than it does.
        3. For each `MatchShape(v: T, e, shape_pattern)`, where `T` is an optional type annotation, let the checked type of `e` be `T'`. 
            1. If `T'` is `ShapeType`, then emit an error if `T` is not a supertype of `ShapeType`. Add `v` to `Γ` with type `T`.
            2. If `T'` is `DynTensorType`:
                1. If the `ndim` of `T'` is `n` ≥ 0, then emit an error if the length of the given `shape_pattern` is not `n`. Let the datatype of `T'` be `d`.
                2. If `T` is not a supertype of `DynTensorType(ndim=len(shape_pattern), dtype=d)`, then emit an error. If `T` is a subtype of that type, emit an error and request a cast.
                3. Add `v` to `Γ` with type `T`.
            3. If `T'` is `ObjectType`, then the only type we can conclude for `v` is `ObjectType`. If `T` is not `ObjectType`, emit an error and request a cast.
            4. If `T'` is `TupleType` or `FuncType`, emit a type error.
    2. If the current block is a `DataflowBlock`, remove any `DataflowVar`s from `Γ` after we have finished processing the bindings in that block.
    3. Finally, the type of the `SeqExpr` is the type of `body` checked under the current `Γ`. Afterwards, remove all bindings added from the `b0`, `b1`, …, `bn` from `Γ`.
11. Let us consider a function `Function(v1 : T1, v2 : T2, ..., vn : Tn, attrs=a) -> Tr: body`.
    1. For handling recursive calls: If the function has been bound to a name `fv` (which may be a `Var` or a `GlobalVar`), then add `fv` to `Γ` with the type `FuncType([T1, T2, ..., Tn], Tr, pure=p)` and proceed as below, where `p` is `True` if a `pure` attribute is included and `False` otherwise. Remove `fv` from `Γ` before returning.
    2. Add `v1` to `Γ` with type `T1`, `v2` with type `T2`, …, and `tn` with type `Tn`. Recursively type check `body` in this new context:
        1. «Determining purity: If `body` contains any call to a function whose return type does not specify that it is pure, a call to an `ExternFunc` that does not have the `pure` attribute, or a call to an `Op` that does not specify the `pure` attribute, then we consider the function to be (potentially) impure. If all calls are to functions whose return type specifies purity or that include the `pure` attribute on the call or `Op`, then the function is treated as pure.»
        2. «Suppose the purity defined in the previous step is `p'`. Suppose the annotated function purity (in the attributes) is `p`. If `p'` is false while `p` is true, then it is a type error; if `p` was omitted, use `p'` for `p`.»
        3. «If the function has the attribute "`force_pure`," then consider `p` to be true, even if the check above judged the function not to be pure. The compiler may emit a warning in this situation.»
        4. Suppose the result of type-checking `body` is `Tr'`. If the current function is not recursive or a mutually recursive global function and `Tr` was omitted, consider the function to have type `FuncType([T1, T2, ..., Tn], Tr', pure=p)`. If `Tr' <: Tr`, then we consider the function to have type `FuncType([T1, T2, ..., Tn], Tr, pure=p)`. «If `Tr <: Tr'`, return an error and require an explicit cast in the function body.» If `Tr'` is not a subtype of `Tr` and `Tr` is also not a subtype of `Tr'` (meaning a dynamic cast cannot succeed), this is an error. 
        5. Remove `v1`, `v2`, …, and `vn` from `Γ` before returning.

# Shapes in Relax

In Relay, shapes are part of tensor types and there is much analysis of tensor shapes done at compile time. In Relax, to allow for greater flexibility for variable-shape tensors and make it easier to implement new operators, shapes can be checked at run time. Though every expression in Relax has a shape associated with it just as expressions also have types, there is no requirement that the shape be expressed at compile time. Instead, the compiler merely requires that an expression's shape define *a way* to compute a fully specified shape at run time. Users have the ability to make use of shape variables and arithmetic expressions to encode a wide variety of shape constraints that can be checked dynamically.

Nevertheless, in many cases, these shapes can be analyzed at compile time (particularly when they are consist of constants or deducible variables) to facilitate compile-time optimization much like is possible with Relay or TIR. Through constant propagation, function inlining, and other partial evaluation–like transformations, we can potentially eliminate many more dynamic checks by allowing some shape computations to be simplified at compile time.

## Defining Shape Computations

In Relax, each expression has an associated shape computation, which defines how that expression's shape can be computed based on the shapes of its subexpressions. We will refer to this computation as `shape_`, as that is what it is called in the implementation. This essentially serves as a mechanism for propagating shape annotations on variable bindings and function definitions to other expressions and enable more compile-time analysis of shapes. In particular, `shape_` is useful for memory planning. These computations can also be used to simplify shape checking and eliminate many dynamic checks. 

### Expressing Dimensions

A tensor shape is a tuple of TIR `PrimExpr`s, where each `PrimExpr` corresponds to a dimension. The use of TIR `PrimExpr`s for shape dimension allows shape computations to express complex constraints that include variables and integer arithmetic expressions in addition to just constant dimensions.

**Scope of Shape Variables**

Shape variables can be introduced in two places in a Relax program: In a function signature, where they may be included with the argument shapes and return shape annotations, or in `MatchShape` bindings. Shape variables used in the function signature are scoped to the entire function in which they appear. Shape variables used in `MatchShape` bindings are scoped only to the `SeqExpr` in which they appear.

**Informal Semantics of `PrimExpr`s for Dimensions**

1. Shape variables can be bound to a value exactly once: at the start of a function for shape annotations on function arguments, in `MatchShape` bindings, or before a function returns (for shape variables on the return type). In particular, matching a `PrimExpr` consisting only of an uninitialized shape variable is treated as its binding (see below on `MatchShape`). After a shape variable has been bound for the first time, future uses of it will refer to the same value.
2. It is not legal to use a shape var that has not yet been bound. This results in an error at run time, though most cases can be detected at compile time.
3. «Local functions will "capture" defined shape variables from the parent scope with their present values in the resulting closure.»
4. If all variables in the `PrimExpr` are defined, `PrimExpr` arithmetic will generally be evaluated according to the semantics of TIR.

### Evaluating `MatchShape`

`MatchShape` allows for binding shape variables in Relax. It can be used with either tensor values or shape values, and in both cases the evaluation of the `PrimExpr`s proceeds similarly.

1. Evaluating `MatchShape(v, t, s)`, where `t` is a tensor value and `s` is a list of `PrimExpr`s corresponding to shape dimensions:
    1. Suppose `s` is `(p1, p2, ..., pn)` , where each variables is a `PrimExpr`. We evaluate `p1`, then `p2`, and so, in that order according to the following rules (corresponding to the `i`th dimension):
        1. If the current `PrimExpr` consists only of an uninitialized shape variable, we bind the shape variable in that scope to the concrete value of the `i`th dimension of the value of `t`.
        2. Evaluate the current `PrimExpr` and compare it to the concrete value of the `i`th dimension of `t`. Raise an error if they do not match.
    2. If `v` is provided, bind `t` to `v` (see the general semantics for how that should be implemented).
2. Evaluating `MatchShape(v, S, s)`, where `S` is a shape value proceeds identically to the above, except the `PrimExpr`s are compared to the `i`th element of `S`.

### General Shape Computation Grammar

Shape computations can consist of the following expressions, which are a subset of general Relax `Expr`s:

```
ShapeCompExpr ::= ShapeExpr(dims: [PrimExpr])
                | RuntimeDepShape()
                | «Tuple(fields: [ShapeCompExpr])»
                | Call(op: Op|ExternFunc, args: [Var|Constant])
                | «TupleGetItem(tuple_value: ShapeCompExpr, index: int)»
```

The shape expressions can be interpreted as follows:

- `ShapeExpr` describes the shape of a tensor as a list of dimensions
- «`Tuple` describes the shapes of each member of a tuple»
- «`TupleGetItem` describes the shape of a member of a tuple»
- `Call` describes the shape of a function (or operator) call return value in terms of its arguments
- `RuntimeDepShape` describes shapes that are unknown at compile time (like when a shape annotation is omitted) or the shapes of values that don't have shapes (like shapes themselves, paradoxically: they *are* shapes but do not *have* shapes).

The `PrimExpr`s in a `ShapeCompExpr` can reference the same shape variables as in shape annotations, with the same semantics.

**Restrictions**

Shape computations are allowed to include calls to operators and even `PackedFunc`s, but these operators and `PackedFunc`s *must* be pure. Shape computations are primarily used for memory planning and it is at the compiler's discretion when, if ever, to evaluate them (except as described below), hence they must not have side effects.

**Shape Annotations**

For shape annotations, we use `ShapeCompExpr` as the grammar, as with `shape_` expressions. `ShapeExpr` is used to annotate shapes of tensor values, «`Tuple` is used to annotate the shapes of tuple values», and `RuntimeDepShape` is used to indicate annotations that have been omitted or shapes that cannot be known at compile time (like the shapes of tensors whose rank is unknown at compile time). `Call` is used to annotate the shapes of calls to operators and «`TupleGetItem` annotates the shapes of tuple indices.»

«For example, suppose we have a tuple where some fields are tensors like the following:

```python
x : Tuple(Tensor((m, n), "int32"), Tuple(), Tensor((), "int32"), Tensor(_, "int32")) = ...
```

It has the shape annotation

```python
Tuple([ShapeExpr([m, n]), Tuple([]), ShapeExpr([]), RuntimeDepShape])
```
»

Note that it is [a well-formedness requirement](https://www.notion.so/Informal-Relax-Language-Specification-d1fdedb8fae84f0d82b9f880f25e7370) that if any field in a type has a `ShapeExpr` annotation, it must be a `DynTensorType` with an `ndim` matching the number of dimensions in the `ShapeExpr`. For example, in the above function signatures, the `ndim` in the type annotations must be 2.

### «Assigning Shape Variables at the Start and End of a Function»

«Shape variables are bound at the start and end of a function or in `MatchShape` bindings. We can describe the behavior at the start and end of a function in terms of the semantics of `MatchShape`, as the shape annotations in function arguments and return types are treated as "syntactic sugar" for `MatchShape` bindings. Suppose a function has the following signature, where the `Ti` are type annotation and the `Si` are shape annotations:

```python
def f(arg1 : (T1, S1), arg2 : (T2, S2), ..., argn : (Tn, Sn)) -> (Tr, Sr): 
    return body
```

This can be treated as a macro that expands to

```python
def f(arg1 : T1, arg2 : T2, ..., argn : Tn) -> Tr:
    check_annotation(arg1, T1, S1)
    check_annotation(arg2, T2, S2)
    ...
    check_annotation(argn, Tn, Sn)
    ret_var = body
    check_annotation(ret_var, Tr, Sr)
    return ret_var
```
»

Because `MatchShape` is defined only for tensor and shape values, we must use a macro to handle other possible types that may be passed into a function, given here in pseudocode:

```python
def check_annotation(e: Expr, s: ShapeCompExpr) -> Expr:
    if s is a ShapeExpr:
        tmp = fresh_var()
        # type checking should ensure that e is always a tensor
        return SeqExpr(
            [BindingBlock([MatchShape(tmp, e, s.dims)])],
            tmp
        )
    «else if s is a Tuple:
        # type checking should ensure that e is always a tuple and the lengths match
        shapes = s.fields
        tmp = fresh_var()
        return SeqExpr(
            [BindingBlock([
                VarBinding(tmp, e),
                # recursive in case we have nested tuples
                VarBinding(fresh_var(), check_annotation(TupleGetItem(tmp, 0), shapes[0])),
                VarBinding(fresh_var(), check_annotation(TupleGetItem(tmp, 1), shapes[1])),
                ...,
                VarBinding(fresh_var(), check_annotation(TupleGetItem(tmp, n-1), shapes[n-1]))
             ])], tmp
        )»
   else if s is a Call:
       tmp = fresh_var()
       return SeqExpr(
           [BindingBlock([
               VarBinding(tmp, e),
               # completely dynamic check that does not assign shape vars.
               VarBinding(fresh_var(), dynamically_check_shapes(shape_of(tmp), s))
           ])], tmp
       )
   «else if s is TupleGetItem:
       val = s.tuple_value
       if val is Tuple:
           return check_annotation(e, val.fields[s.index])
       # otherwise, evaluate it
       return SeqExpr(
          [BindingBlock([
              VarBinding(tmp, e),
              VarBinding(fresh_var(), dynamically_check_shapes(shape_of(tmp), s))
          ])], tmp
       )»
   else if s is RuntimeDepShape:
       # no need to check
       return e
```

### Evaluating Shape Expressions

Every shape expression in the program (`shape_`) is associated with a program expression. Other than in the above procedure for checking function parameter shapes and the return shape, the specification does not guarantee that any `shape_` expression will ever be evaluated or how many times it may be evaluated; `shape_` is intended primarily for the benefit of memory planning. Hence, all `shape_` expressions must be pure and must be guaranteed to terminate. The `shape_` for a given expression `e` is intended to be evaluated *before* `e`.

Shape expressions follow the same evaluation rules as general program expressions. In particular, shape functions are permitted to reference any variable that is in scope at the point of its associated expression; i.e., when evaluated, they form closures that capture any free variables (Relax variables and shape variables) referenced in their body. The `RuntimeDepShape` expression has no semantics at run time and indicates a shape that cannot be predicted in advance. If a `RuntimeDepShape` is encountered at any point while dynamically checking a shape match (see the `check_annotation` procedure above), it should "short-circuit" the match and cause the match to succeed immediately.

### Building Up `shape_` for Each Expression

For each expression type, we can recursively build up an associated `shape_` expression according to the following rules:

1. For `Constant(value)`, the `shape_` expression is a `ShapeExpr` corresponding to the concrete shape of `value`. For example, for `Constant(1)`, `shape_` is `ShapeExpr([])` and for `Constant([1, 2])`, `shape_` is `ShapeExpr([2])`.
2. «For `Tuple(fields)`, `shape_` can be defined as `Tuple([field.shape_ for field in fields])`.»
3. For `ShapeExpr`s, `shape_` is `RuntimeDepShape`.
4. `RuntimeDepShape` expressions should appear only in shape expressions; their `shape_` is not defined.
5. For `If(cond, true_branch, false_branch)`, we compare the `shape_` of `true_branch` and `false_branch`. If these can be proven equivalent (by a method that the compiler implementation is free to determine), then the `If` node's `shape_` is that shape. If they do not match, then we set it to `RuntimeDepShape`.
6. For `SeqExpr`, we set the `shape_` to be the `shape_` of the body expression. The `shape_` must respect the scoping rules for the `SeqExpr`: If the `shape_` of the body expression contains shape variables not defined in the outer scope (i.e., shape variables that are scoped to the `SeqExpr` only) or if the `shape_` contains any `Var`s or `DataflowVar`s scoped to the `SeqExpr`, use `RuntimeDepShape` as the shape.
7. For handling variable bindings:
    1. For the arguments to a function, set the `shape_` to the annotated shape. If the annotation is omitted, use `RuntimeDepShape`.
    2. In the general `VarBinding(v, e)`, if `v` does not have a shape annotation or the annotation is `RuntimeDepShape`, then we set the `shape_` of `v` to the `shape_` of `e`. If `v` has a shape annotation, then if the `shape_` of `e` can be proven equivalent to the shape annotation, use the shape annotation for the `shape_` of `v`. «Otherwise, give an error and require an explicit `MatchShape`.»
        
        It is up to the compiler implementation to decide what method to use for attempting to prove equivalence.
        
    3. For bindings where the RHS is a function literal or assigning the `shape_` of a `GlobalVar`, see the rule for `Function` nodes.
    4. For `MatchShape(var, value, shape)`, we set the `shape_` of `var` to `shape`, as it will be dynamically checked.
8. «For `TupleGetItem(tuple_value, i)`, we examine the `shape_` of `tuple_value`; suppose it is `s`. If `s` is a `Tuple`, then we use its `i`th field. If it is `RuntimeDepShape`, we use `RuntimeDepShape`. If it is a `Call` to a function that returns a tuple with at least `i + 1` members, set the `shape_` to `TupleGetItem(s, i)`. Otherwise, raise an error at compile time (though this should not happen if type checking has passed).»
9. For `Call` nodes:
    1. For a call to an `ExternFunc`, we use `RuntimeDepShape` because we cannot analyze the shapes of arbitrary `PackedFunc`s and must check dynamically.
    2. For a call to an `Op`, we use the manually defined `FInferShape` macro if it has been defined and `RuntimeDepShape` if it has not. `FInferShape` is a function that takes in the call node and produces a `ShapeCompExpr`.
    3. «For all other cases with `Call(op, args)`, we consider the following cases:
        1. If `op` is a `GlobalVar` or a `Var` that refers to a function defined in the current scope, look up the `Function` node it references; let us call it `f`. Similarly, if `op` is itself a `Function` node, let `f` be `op`.
            
            Attempt to perform [beta-reduction](https://en.wikipedia.org/wiki/Lambda_calculus#%CE%B2-reduction) on `f`'s return shape. A pseudocode procedure for this beta-reduction is given below, as a macro.
            
            1. If the return shape of `f` is a `Call` node or contains any `Call` nodes, substitute any parameters of `f` for the corresponding member of `args`. (E.g., if `f` has parameters `p1`, `p2`, …, `pn` and any of these variables appears in the return shape, `p1` should be replaced with the first member of `args`; `p2`, with the second; etc.) If any member of `args` that is substituted this way is not a `Var` or `Constant`, consider beta-reduction to fail.
            2. For each shape annotation in the parameters of `f`, attempt to match it with the `shape_` of the corresponding member of `args`, substituting shape variables in the return shape accordingly. If the `shape_` of the member of `args` is `RuntimeDepShape`, consider beta-reduction to fail. If the `shape_` is not `RuntimeDepShape` but is incompatible with the parameter's shape annotation (e.g., a `Tuple` where a `ShapeExpr` was expected), report an error at compile time.
            
            If `f`'s return shape is `RuntimeDepShape`, then consider the call result to have `RuntimeDepShape`. If beta-reduction is considered to fail, then consider the call result to have `RuntimeDepShape`. If it succeeds, use the resulting shape as the `shape_` of the call result.
            
        2. Otherwise, consider the result of the call to have `RuntimeDepShape`.
	»
10. For a function node, set the `shape_` to `RuntimeDepShape`.

### Procedure for Substituting a Function Return Shape to Determine the Shape of a Call

The `substitute_shape` procedure defined below describes how the shape expression for a call result can be defined given the call arguments and the return shape annotation on the corresponding function node. Note that this procedure can obtain much more precise results in the cases of `Call` or `TupleGetItem` return shapes.

```python
def map_shape_vars(param_shape: ShapeCompExpr, arg_shape: ShapeCompExpr, shape_var_mapping: {tir::Var : PrimExpr}) -> bool:
    if param_shape is RuntimeDepShape or arg_shape is RuntimeDepShape:
        return False
    if param_shape is ShapeExpr and arg_shape is ShapeExpr:
        if len(param_shape.values) != len(arg_shape.values):
            raise UnificationError("Shapes are of incompatible ranks")
        for param_dim, arg_dim in zip(param_shape.values, arg_shape.values):
            if param_dim in shape_var_mapping:
               # syntactic equality
               if arg_dim != shape_var_mapping[param_dim]:
                   # if they are statically not equal, e.g., 5 != 7 or 3 + 3 != 3*3
                   if can_prove_not_equal(arg_dim, shape_var_mapping[param_dim]):
                       raise UnificationError("Incompatible dimensions")
                   else:
                       return False
            else:
                shape_var_mapping[param_dim] = arg_dim
        return True
    if param_shape is Tuple and arg_shape is Tuple:
        if len(param_shape.fields) != len(arg_shape.fields):
            raise UnificationError("Tuples are of incompatible lengths")
        for param_field, arg_field in zip(param_shape.fields, arg_shape.fields):
            ret = map_shape_vars(param_field, arg_field, shape_var_mapping)
            if not ret:
                return False
        return True
    if param_shape is TupleGetItem and arg_shape is TupleGetItem:
        # Does not necessarily indicate a unification error, 
        # depending on what the tuple values are.
        # Constant folding the TupleGetItem nodes could improve this unification case
        if param_shape.index != arg_shape.index:
            return False
        return map_shape_vars(param_shape.tup_value, arg_shape.tup_value)
    if param_shape is Call and arg_shape is Call:
        # no dimension mapping to do in this case
        return True   
    # if either is a Call or TupleGetItem, it is possible that the shapes 
    # can match dynamically even if they don't match statically
    if (param_shape is Call
        or param_shape is TupleGetItem
        or arg_shape is Call
        or arg_shape is TupleGetItem):
        return False
    raise UnificationError("Incompatible shape constructs")
  
def substitute_vars(target: Expr, var_mapping: {Var: Expr}, shape_var_mapping: {tir::Var: PrimExpr}) -> Expr:
    def substitute_shape_vars(target: PrimExpr):
        if target is tir::Var:
            if target in shape_var_mapping:
                return shape_var_mapping[target]
            else:
                return target
        # proceed recursively in all subexpressions, checking for vars

    if target is Var:
        if target in var_mapping:
            return var_mapping[target]
        return target
    if target is ShapeExpr:
        return ShapeExpr([
            substitute_shape_vars(dim) 
            for dim in target.values
        ])
    # recurse through all other cases, checking for vars and shape exprs analogously

def substitute_shape(func_params, arg_exprs, ret_shape):
    var_mapping = {param: arg_expr for param, arg_expr in zip(func_params, arg_exprs)}
    shape_var_mapping = {}
    for param, arg_expr in zip(func_params, arg_exprs):
        can_unify = map_shape_vars(param.shape_, arg_expr.shape_, shape_var_mapping)
        if not can_unify:
            return RuntimeDepShape()
    
    new_shape = substitute_vars(ret_shape, var_mapping, shape_var_mapping)
    if new_shape contains any free (Relax or shape) variables:
        return RuntimeDepShape()
    return new_shape
```

### Note on Proving Shapes Equivalent and Eliminating Dynamic Checks

There can be some complexity involved in checking whether two shapes match during shape inference. A very simple, conservative method for determining equality is simply using alpha-equivalence: If the two shapes have the same structure, then they are equivalent. However, this method is conservative and can overlook numerical properties in `PrimExpr`s. We leave it up to compiler implementations as to whether to use more advanced methods for proving equivalence, such as attempting to use algebraic rewrite rules. (As a consequence, portability requires inserting dynamic checks wherever there needs to be a comparison of shapes.)

Note that optimizations like function inlining or constant folding could allow for simplifying many shape annotations and expressions and make it possible to conclude at compile time that shapes in more cases are equivalent. In general, developing compiler infrastructure for partial evaluation and reasoning about common situations with shape annotations may eliminate many dynamic checks.

Applying some kind of normalization or algebraic simplifications to `PrimExpr`s used in shape annotations and in `shape_` fields can also make it easier to conclude that certain dynamic checks may not be necessary by increasing the likelihood that more `shape_` expressions could be made syntactically identical to the shape annotations. It would also be possible to generate compile-time warnings if analysis reveals that two shapes may not match (either using rewrite rules or by trying random values for shape variables and checking).

Since most dynamic shape checks are done for safety, it may be feasible to introduce a compilation mode that eliminates almost all dynamic shape checks. Some shape checks may not be possible to eliminate, since the body of the program may construct `ShapeExpr`s and use them in calls to `PackedFunc`s, so some bindings to shape variables may need to be preserved, per a liveness analysis.

## Possible Extensions to the Shape Expression System

We may consider two possible extensions to the shape expression system in order to accommodate two further cases:

1. An explicit wildcard dimension (e.g., using `tir::Any`) to allow for dimensions to be specified as "unknown" in function return shapes. As described at present, the only way for a function to specify a partly unknown return shape is to make the entire return shape unknown (`RuntimeDepShape`), which loses partial shape information.
2. Adding `shape_` expressions consisting of functions, to allow arbitrary closures to have a known shape. This would allow the shapes of calls to closures of unknown origin (namely, in a higher-order function) to have their shapes correctly inferred rather than made `RuntimeDepShape`.

In both cases, these additions would entail additional complexity (shape inference macros for operators would have to deal with potential `tir::Any` nodes and we would have to define rules for constructing, calling, and simplifying functions in `shape_` expressions). However, the advantage of implementing these features would be increasing the amount of shape information present at compile time and hence that could be used by lower levels of the compiler stack. The present defaults of using `RuntimeDepShape` means that either of these changes could be pursued in the future without breaking existing code, since these would generally have to be paired with explicit `MatchShape` dynamic checks, which will still work even if we add rules to automatically infer the shapes in those cases.

# Detailed Semantics

## Program Entry Point

In the `IRModule`, every mapping of a `GlobalVar` to a `Function` node or a TIR `PrimFunc` should be processed first and added to the global scope. «Global functions that have a `global_symbol` attribute should be externally linked, meaning that they can be invoked as program entry points; those that do not have a `global_symbol` attribute can be called only from within the global functions in the `IRModule`.»

The rules for evaluating `Function` nodes into closures are given below. TIR `PrimFunc`s evaluate into objects that are opaque to Relax; these objects have type `Object` and can be used only by the `call_tir` operator. None of the values in global scope is mutable. Execution of a Relax function in an IR module thus begins by evaluating all globally visible functions into a form in which they can be accessed.

## Evaluating Expressions

For each expression, we define how it affects the program's visible state and the order in which they are evaluated. Below, all evaluation results are passed by reference (and hence possibly alias) unless it is explicitly specified that they allocate new values.

1. The node `Constant(value)` creates a new tensor whose contents are `value`.
2. A variable (whether `Var`, `DataflowVar` , or `GlobalVar`) evaluates to the stored value for that variable in the current scope.
3. The node `Tuple([e1, e2, ..., en])` evaluates `e1` (yielding value `v1`), then `e2` (yielding value `v2`), …, and finally `en` (yielding value `vn`) in that order and creates a new tuple value containing `v1`, `v2`, …, and `vn` in that order.
4. The node `TupleGetItem(t, i)` is evaluated by first evaluating `t` (which, per type checking, must evaluate to a tuple) and then returning the `i`th field of the result.
5. The node `ShapeExpr([p1, p2, ..., pn])` evaluates the `PrimExpr`s `p1` (yielding dimension value `v1`), `p2` (yielding dimension value `v2`), …, and finally `pn` (yielding dimension value `vn`) in that order, using the current shape context, and creates a new shape value whose dimensions are `v1`, `v2`, …, `vn`, in that order.
6. `RuntimeDepShape` expressions must not appear in the general body of a program; it is a well-formedness error if they do. They do not have any defined semantics.
7. The node `Function([v1, v2, ..., vn], body)` returns a new closure containing the function definition itself and a mapping of any free Relax variables or shape variables in `body` to the values they hold in the current scope when the `Function` node is encountered. If the function is the RHS of a local binding, the bound variable should also be included in the closure's binding map and should be mapped to the closure itself (to allow for recursive calls). Closure capturing is done *by reference*; no values will be copied and references to captured values will alias their values in the outer scope. `DataflowVar`s are not captured by closures.
8. The node `If(cond, true_branch, false_branch)` is evaluated as follows:
    1. First `cond` is evaluated. Let the result be `r` (per type checking, it must be a bool scalar).
    2. If `r` is true, evaluate the `true_branch` and return its result.
    3. If `r` is false, evaluate the `false_branch` and return its result.
9. The node `Call(op, [arg1, arg2, ..., argn])` is evaluated as follows:
    1. If `op` is an `ExternFunc` node, then evaluate `arg1`, `arg2`, …, `argn` in that order and call the results `a1`, `a2`, …, `an`. Next, look up the `PackedFunc` registered under the global symbol name. If it exists (it is an error at run time if it does not), call the `PackedFunc` using the given arguments and return the result. Note that if a TIR `PrimFunc` in the `IRModule` has a global symbol attribute registered, it can be called as an `ExternFunc` using that global symbol as well. `PackedFunc`s may have arbitrary side effect and are responsible for whether the result is a newly allocated value or aliases another value.
    2. If `op` is an `Op` node, then evaluate `arg1`, `arg2`, …, `argn` in that order and call the results `a1`, `a2`, …, `an`. It is up to the compiler implementation to decide how operators should be implemented (some may have an associated `PackedFunc` and others may be built into the executor implementation). The operator may mutate its arguments. It is also up to the operator implementation as to whether the result is newly allocated or aliases another value. «(TODO: Once we have operators for logical and AND and OR, we should also define short-circuiting semantics for those.)»
    3. In all other cases, first evaluate `op` (it must evaluate to a closure). Next, we evaluate  `arg1`, `arg2`, …, `argn` in that order and call the results `a1`, `a2`, …, `an`. Push a new scope onto the stack where arguments `v1`, `v2`, …, `vn` in the closure are bound to `a1`, `a2`, …, and `an`, respectively, and all variables saved in the closure are added to the scope. Evaluate the closure body in this new scope; this will be the return value of the call. Pop the scope before returning the value.
10. For the node `SeqExpr(blocks, body)`, we evaluate as follows:
    1. Push a new scope onto the stack.
    2. Iterate through the `BindingBlock`s in `blocks` in order. We will call the current one `block`. For each binding in `Block`:
        1. If the binding is `MatchShape(var, value, shape)`, perform the shape matching and shape variable updates as described in the shape evaluation section. If `var` is provided, `var` will be bound to `value` in the current scope; this assignment is aliasing and no new value is allocated. If `var` is not provided, then the shape check is performed and shape variables are updated, but no new binding is introduced.
        2. If the binding is `VarBinding(var, value)`, then evaluate `value` and bind `var` to that value in the current scope; this assignment is aliasing and no new value is allocated.
        3. If `block` is a `DataflowBlock`, remove all `DataflowVar`s bound in the block from the current scope before proceeding to the next block.
    3. After iterating through the binding blocks, evaluate `body` in the current scope. That will be the return value of the `SeqExpr`.
    4. Pop the scope, removing any `Var` bindings introduced in the `SeqExpr`. This should also remove any shape variables introduced and bound in the `SeqExpr` as well.

### Optimizations

Optimizations are allowed to reorder and modify the operations of a program in any way so long as they do not change the value returned by evaluating the program or any visible behavior of the program. For the purposes of compilation, visible behaviors consist of side effects like mutating values in the program or external effects like I/O (printing to the console, creating files, etc.) and the order and number of times in which they happen.

«Within `DataflowBlock`s, it is permitted for the compiler to remove or reorder `MatchShape` or `cast` operations even though this can affect the "visible behavior" of the program (since they can exit with an error). It is also permitted for the compiler to optimize away potential non-termination within `DataflowBlock`s: For example, if some pure function `f` has an integer return type and does not terminate, it is permissible to optimize `f() - f()` to 0 within a `DataflowBlock`. In general, the compiler is permitted to make programs "more defined" (terminating when the original did not terminate, not raising an error when the original raised an error) within a `DataflowBlock`, but never "less defined" (giving an error when the original did not give an error, not terminating when the original did not terminate). Outside of `DataflowBlock`s, error messages and potential non-termination must be preserved faithfully.»

The specification makes no guarantees about certain memory-related properties and hence also does not consider them to be "visible behaviors":

- Whether an allocation happens at a given point. Compiler implementations are permitted to reuse already-allocated memory if it would not interfere with visible state in any other way, per the aliasing rules (`PackedFunc`s or operators may mutate values that are passed to them and those mutations should be visible as per aliasing in this specification). Copying values or sharing representations (e.g., interning constants) between values may be done only if they will not affect any other visible behaviors, dependent on the aliasing behavior.
- It is entirely the domain of compiler implementations to make guarantees (or not) as to whether memory allocations will succeed.
- `PackedFunc`s or operators can, in principle, access information about the machine's state and make changes to allocation policies or the state that affect how memory allocations are performed. The specification makes no guarantees in such an event.

These semantic rules assume a single thread of evaluation on a single host machine. At this time, it is unspecified as to how Relax programs should behave if split over distinct threads or across multiple machines.

### Notable Operators

The above evaluation rules are general, but leave much room for implementations of operators to specify custom semantics. Certain operators are used to perform common operations and will be discussed here as well.

- `call_tir(prim_func, arg1, arg2, ..., argn, shape, type_args=[aT])`: `prim_func` must be a `PrimFunc` object in the current `IRModule` (we will call it `f`). The `shape` argument gives the shapes of the result of calling the TIR `PrimFunc`: It must be either of `ShapeType` (corresponding to returning a single tensor) or `TupleType` whose members are `ShapeType` (corresponding to returning a tuples of tensors). The type arg `aT` gives the type of the result of calling the `PrimFunc` and it must correspond to `shape` (namely, if `shape` is of `ShapeType`, `aT` must be a `DynTensorType`; if `shape` is of `TupleType`, `aT` must be a `TupleType` whose fields are `ShapeType`). `aT` is used especially to provide the `dtype` of returned tensors.
    
    Based on `shape`, the resulting tensor or tuple `r` will be allocated according to the sizes given in `shape`. `f` will be called in destination-passing style, like so: `f(arg1, ..., argn, *r)`. The asterisk denotes that if `r` is a tuple, it will be "unrolled," so the call will be `f(arg1, ..., argn, r1, ..., rn)`, where the `ri` are the fields of `r`. `f` is expected to mutate *only* `r` to give the output of the function, hence `call_tir` is considered pure. If the shape or data type of the actual result do not correspond to `shape` or `aT`, an error is issued.» After the call, `r` is returned.
    
- «`call_dps_packed(global_symbol, arg1, arg2, ..., argn, shape, type_args=[aT])`: Proceeds similarly to `call_tir`, except it calls a `PackedFunc` registered under the name `global_symbol`. The `PackedFunc` may modify `arg1`, `arg2`, …, or `argn` in addition to the result tensor, so purity is not assumed. A type argument `aT` must be given to specify the return type.»
- `shape_of(t)`: Given a tensor argument `t`, it returns its shape. The return value is a new shape object.
- «`cast(v, type_args=[aT])`: Given an argument `v`, it dynamically checks if `v`'s run-time representation is a subtype of `aT`. If it is not, it exits the program with an error message. Otherwise, it returns `v`.»

