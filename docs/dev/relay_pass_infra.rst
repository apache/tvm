..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

.. _relay-pass-infra:

Relay Pass Infrastructure
=========================

Relay features a series of optimization passes which improve performance metrics
of models such as mean inference, memory footprint, or power consumption for
specific devices. There is a suite of standard optimizations as well as machine
learning-specific optimizations including constant folding, dead code
elimination, operator layout alteration, and operator fusion, etc. Each of these
passes is structured as a Relay-to-Relay transformation on the abstract syntax
tree (AST) using the analysis result collected during and/or before traversal.

However, as Relay evolves quickly, the need for a more systematic and efficient
way to manage these passes is becoming apparent. This doc describes the design of
such an infra that takes the advantage of the way production compilers are used to
manage the optimization passes and the style modern deep learning frameworks
adopted to build up layers.

For example, many existing production compilers, such as GCC and LLVM, employ
pass managers to effectively manage the execution of passes. Initially managing
passes is straightforward as the number of passes is small, but mature compilers
will contain hundreds of individual passes. Often external users will want to
have custom passes correctly scheduled without having to modify a single
handcrafted pass order.

Similarly, modern deep learning frameworks, such as Pytorch and MXNet
Gluon, also have the tendency to enable pass-style layer construction
scheme through `Sequential`_ and `Block`_, respectively. With such constructs,
these modern frameworks are able to conveniently add modules/layers to their
containers and build up neural networks easily.

The design of the Relay pass infra is largely inspired by the the hierarchical
pass manager used in LLVM and the block-style containers used in the popular
deep learning frameworks. The major goals of the pass infra include:

#) enabling better programmatic orchestration of optimizations. This allows
   users to flexibly customize and build their own optimization pipelines.

#) providing a user-friendly way to debug optimization passes.

#) alleviating developers from manually and respectively resolving the
   dependencies between passes.

#) simplifying the implementation of new passes for developers. For example, we
   allow users to implement a pass in Python and let the pass infra manipulate
   its execution.

The Design
----------

We focus on ease of extension for users, making it possible for users to quickly
add new passes without loss of backward compatibility. The design contains both
the backend and the frontend. The former implements the main logic of the pass
infra. The latter provides simple APIs for users to interact with, i.e.,
allowing users to quickly create their own optimization pipelines.

C++ Backend
~~~~~~~~~~~

We provide a ``PassInfo`` object to contain the basic information needed by
a pass. ``name`` is the pass name, ``opt_level`` indicates at which optimization
level the pass will be enabled, and ``required`` represents the passes that are
required to execute a certain pass (see `include/tvm/ir/transform.h`_ for
more details). For example, during registration of a pass (will be covered in
later), the pass developers can specify the name of the pass, the optimization
level it will be performed at, and/or the passes that are required.
``opt_level`` could be used to help the pass infra identify if a certain pass
needs to be executed when running under a user-provided optimization level. The
``required`` field can be used by the pass infra to resolve pass dependencies.

.. code:: c++

    class PassInfoNode : public RelayNode {
      std::string name;
      int opt_level;
      std::vector<std::string> required;
    };

PassContext
^^^^^^^^^^^

``PassContext`` carries useful information for an optimization pass. For
example, it contains the error reporting system so optimization authors can
provide diagnostics about why an optimization fails. ``PassContext`` is also
designed to replace the old ``BuildConfig`` which was used to help users
configure the compilation options, including optimization level and
required/disabled passes, etc. For instance, we may have a configuration which
performs all passes at ``opt_level=3`` with some disabled passes using
``disabled_pass=xx`` provided by ``PassContext``. Now we could glob all passes
at ``opt_level=3`` and exclude those in the disabled pass list.

This class is designed for users to conveniently write the Python ``with``
syntax to perform optimizations under a certain configuration. In addition, the
users can obtain the context that is available within a certain program scope in
a thread-safe way through ``PassContext::Current()``, since a thread-local store
``RelayPassContextThreadLocalStore`` is used to hold the created pass context
objects. Examples will be provided later to show how we can use both the C++ and
Python APIs to create a compilation pipeline using pass context.

.. code:: c++

    class PassContextNode : public RelayNode {
     public:
      ErrorReporter err_reporter;
      int opt_level{2};
      int fallback_device{static_cast<int>(kDLCPU)};
      tvm::Array<tvm::Expr> required_pass;
      tvm::Array<tvm::Expr> disabled_pass;
    };

    class PassContext : public NodeRef {
     public:
      TVM_DLL static PassContext Create();
      TVM_DLL static PassContext Current();
      /* Other fields are omitted. */

     private:
      // The entry of a pass context scope.
      TVM_DLL void EnterWithScope();
      // The exit of a pass context scope.
      TVM_DLL void ExitWithScope();

      // Classes to get the Python `with` like syntax.
      friend class tvm::With<PassContext>;
    };

    struct RelayPassContextThreadLocalEntry {
      /*! \brief The default pass context. */
      PassContext default_context;
      /*! \brief The current pass context. */
      std::stack<PassContext> context_stack;
      RelayPassContextThreadLocalEntry() {
        default_context = PassContext(make_node<PassContextNode>());
      }
    };

    /*! \brief The thread-local store to hold the pass context. */
    typedef dmlc::ThreadLocalStore<RelayPassContextThreadLocalEntry>
         RelayPassContextThreadLocalStore;

Pass Constructs
^^^^^^^^^^^^^^^

The pass infra is designed in a hierarchical manner, and it could work at
different granularities of Relay programs. A pure virtual class ``PassNode`` is
introduced to serve as the base of the different optimization passes. This class
contains several virtual methods that must be implemented by the
subclasses at the level of modules, functions, or sequences of passes..

.. code:: c++

    class PassNode : RelayNode {
      virtual PassInfo Info() const = 0;
      virtual Module operator()(const IRModule& mod
                                const PassContext& pass_ctx) const = 0;
    };

The functor shows how a pass must be realized, i.e. it always works on a
:py:class:`IRModule` under a certain context. All passes are designed in a ``Module`` to ``Module``
manner. Therefore, optimizations governed by the pass infra will
always update the whole module.

Several subclasses have been created to implement different types of
optimization passes, e.g., function-level passes, module-level passes, and
sequential passes.  Each subclass itself could act as a pass manager. For
instance, they could collect the required passes and execute them or build
a dependency graph based on the given metadata. The full definition of them
can be found in `src/relay/ir/transform.cc`_ and `src/ir/transform.cc`_.

Module-Level Passes
^^^^^^^^^^^^^^^^^^^

Module level passes are geared mainly for global and inter-procedural
optimizations (IPO), which are similar to the module pass used in LLVM. Some
typical passes in Relay that need the global picture of a module, such as
A-normal form conversion and lambda lifting, etc., fall into this set. At this
level, users can even add and/or delete functions in a module.

.. code:: c++

    class ModulePassNode : PassNode {
      PassInfo pass_info;
      runtime::TypedPackedFunc<Module(Module, PassContext)> pass_func;
      Module operator()(const Module& mod, const PassContext& pass_ctx) const final;
      // Other members/methods are omitted
    };

``pass_info`` maintains the information needed by a module-level pass.
``pass_func`` sketches the real optimization. For example, we may need to
perform dead code elimination on the module. We could implement the algorithm in
the ``pass_func`` and let it run on a module. It will then remove the dead code
including the unused functions in the module. Note that this field is designed
as a packed function, which enables the implementation of the optimization in
both C++ and Python.

Function-Level Passes
^^^^^^^^^^^^^^^^^^^^^

Function-level passes are used to implement various intra-function level
optimizations for a given Relay module. It fetches one function at a time from
the function list of a module for optimization and yields a rewritten Relay
function. Most of Relay's passes can be classified into this category, such as
common subexpression elimination and inference simplification, etc.

Note that the scope of passes at this level is a Relay function. Therefore, we
cannot add or delete a function through these passes as they are not aware of
the global information.

.. code:: c++

    class FunctionPassNode : PassNode {
      PassInfo pass_info;
      runtime::TypedPackedFunc<Function(Function, Module, PassContext)> pass_func;
      Module operator()(const Module& mod, const PassContext& pass_ctx) const final;
      bool SkipFunction(const Function& func) const;
      // Other members/methods are omitted...
    };

``pass_info`` is identical to what we just described in the module pass.
``pass_func`` takes a function for optimization, it also needs a module as we
may use it for reporting errors. A function could be annotated with
"SkipOptimization" so that it will be ignored during optimization.

Sequential Passes
^^^^^^^^^^^^^^^^^

``SequentialPass`` is similar to Pytorch ``nn.Sequential`` that contains a host
of passes for execution.

.. code:: c++

    class SequentialPassNode : PassNode {
      PassInfo pass_info;
      // Passes need to be executed.
      Array<Pass> passes;
      bool PassEnabled(const PassInfo& info) const;
      Module operator()(const Module& mod, const PassContext& pass_ctx) const final;
    };

Only a few passes currently in Relay are put in this group. For example,
``FoldScaleAxis`` requires to dispatch ``ForwardFoldScaleAxis`` and
``BackwardFoldScaleAxis`` internally. In addition, ``BackwardFoldScaleAxis`` is
recommended to be fulfilled first. This pass, hence, is an ideal candidate for
``SequentialPass``.

The following code shows how individual passes in a sequential pass are invoked.
Essentially, we sequentially execute each pass in a sequential pass using the
order that they were appended to the pass list.

.. code:: c++

    Module SequentialNode::operator()(const Module& module,
                                      const PassContext& pass_ctx) const {
      Module mod = module;
      for (const Pass& pass : passes) {
        CHECK(pass.defined()) << "Found undefined pass for optimization.";
        const PassInfo& pass_info = pass->Info();
        if (!PassEnabled(pass_info))  continue;
        for (const auto& it : pass_info->required) {
          const auto* name = it.as<tvm::ir::StringImm>();
          CHECK(name);
          mod = GetPass(name->value)(mod, pass_ctx);
        }
        mod = pass(mod, pass_ctx);
      }
      return mod;
    }

Upon the invocation of a pass, we first check if this pass is enabled. This is
done by first checking if the pass is explicitly disabled by a user, followed by
inspecting if it is specified as a required pass by the user. If it is still
undetermined whether this pass is enabled, its ``opt_level`` will be checked.
This pass will be enabled and therefore executed only when its optimization
level is not less than the configured optimization level in the pass context.

To execute the pass, we need first to retrieve the registered pass in the TVM
packed function registry using the pass name. This is possible because every
pass is registered with an API endpoint as we will show later.

.. code:: c++

    Pass GetPass(const std::string& pass_name) {
      using tvm::runtime::Registry;
      std::string fpass_name = "relay._transform." + pass_name;
      const auto* f = Registry::Get(fpass_name);
      CHECK(f != nullptr) << "Cannot find " << fpass_name
                          << "to create the pass " << pass_name;
      return (*f)();
    }

Some helper functions are provided to create each type of these aforementioned
passes. These helpers are also exposed to the Python frontend for users to
favorably use Python APIs to create a specific pass object.

.. code:: c++

    FunctionPass CreateFunctionPass(std::string name,
                                    int opt_level,
                                    PassFunc pass_func);

    ModulePass CreateModulePass(std::string name,
                                int opt_level,
                                PassFunc pass_func);

    SequentialPass CreateSequentialPass(std::string name,
                                        int opt_level,
                                        Array<Pass> passes,
                                        Array<tvm::Expr> disabled);

C++ Sequential Example
^^^^^^^^^^^^^^^^^^^^^^

Let's now take an example to illustrate how the pass infra works on
``SequentialPass``. For illustrative purpose, only a code snippet is provided.
First, we create a simple Relay program, ``y = f(x)``. Then, we build a module
based on the function. After creating the module, we instantiate a sequential
pass object which contains some standard Relay optimization passes, including
type inference, dead code elimination, common subexpression elimination, and
layout alteration.

Finally, a pass context is constructed and the passes will be executed
sequentially. During the execution of these passes, the pass dependency will be
resolved automatically as we have encoded the dependent passes during
registration.

.. code:: c++

    // Create a simple Relay program.
    auto tensor_type = relay::TensorTypeNode::make({}, tvm::Bool());
    auto x = relay::VarNode::make("x", relay::Type());
    auto f = relay::FunctionNode::make(tvm::Array<relay::Var>{ x }, x, relay::Type(), {});

    auto y = relay::VarNode::make("y", tensor_type);
    auto call = relay::CallNode::make(f, tvm::Array<relay::Expr>{ y });
    auto fx = relay::FunctionNode::make(tvm::Array<relay::Var>{ y }, call, relay::Type(), {});

    // Create a module for optimization.
    auto mod = IRModule::FromExpr(fx);

    // Create a sequential pass.
    tvm::Array<relay::transform::Pass> pass_seqs{
       relay::transform::InferType(),
       relay::transform::DeadCodeElimination(),
       relay::transform::EliminateCommonSubexpr(),
       relay::transform::AlterOpLayout()
    };
    relay::transform::Pass seq = relay::transform::Sequential(pass_seqs);

    // Create a pass context for the optimization.
    auto ctx = relay::transform::PassContext::Create();
    ctx->opt_level = 2;
    ctx->fallback_device = kDLCPU;

    // Use the Python with syntax to execute the sequence of optimizations.
    tvm::With<relay::transform::PassContext> scope(ctx);
    mod = seq(mod);

    // View the updated module.
    LOG(INFO) << relay::AsText(mod) << std::endl;

Other types of passes should be directly invoked for execution on a module. For
example, users can directly apply const folding pass on a given module, ``mod
= transform::FoldConstant()(mod)``. However, it is users' responsibility to
execute the required passes explicitly.

Pass Registration
~~~~~~~~~~~~~~~~~

We've covered the concept of different level of passes and the context used for
compilation. It would be interesting to see how easily users can register
a pass.  Let's take const folding as an example. This pass has already been
implemented to fold constants in a Relay function (found in
`src/relay/pass/fold_constant.cc`_).

An API was provided to perform the ``Expr`` to ``Expr`` transformation.

.. code:: c++

    Expr FoldConstant(const Expr& expr);

In order to register this pass to the pass infra, we first need to decide at
which level this pass will be performed. As const folding happens on individual
functions, we should intuitively create a ``FunctionPass`` for it through
``CreateFunctionPass``. The ``pass_func`` is returned as a packed function that
invokes the ``Expr`` to ``Expr`` API on each function in a Relay module. ``{}``
indicates that no prerequisite is required for this pass. Otherwise, the pass
developer has to identify and list them.

Meanwhile, a pass API endpoint is registered with the name
``relay._transform.FoldConstant``. This pass, therefore, becomes an entry in the
registry that can be accessed by both C++ (e.g. the ``GetPass`` above) and
Python when needed.

.. code:: c++

    namespace transform {

    Pass FoldConstant() {
      runtime::TypedPackedFunc<Function(Function, Module, PassContext)> pass_func =
        [=](Function f, Module m, PassContext pc) {
          return Downcast<Function>(FoldConstant(f));
      };
      return CreateFunctionPass(pass_func, 2, "FoldConstant", {});
    }

    TVM_REGISTER_GLOBAL("relay._transform.FoldConstant")
    .set_body_typed(FoldConstant);

    }  // namespace transform

To allow other C++ modules to apply this pass, we declare a free function in
`include/tvm/relay/transform.h`_ as the following:

.. code:: c++

    TVM_DLL Pass FoldConstant();

Python Frontend
~~~~~~~~~~~~~~~

Only some simple APIs are needed for the frontend side. For example, we can
provide users the following APIs to create and execute a pass (full
implementation is provided in `python/tvm/relay/transform.py`_). The backend
receives the information and decides which function it should use to create
a Pass object.

PassContext
^^^^^^^^^^^

Python frontend provides a wrapper for the ``PassContext`` to enable the
``with`` syntax by overriding ``__enter__`` and ``__exit__``. A ``current``
static method is offered for users to get the context that is in use under
a certain scope.

.. code:: python

    @register_relay_node
    class PassContext(RelayNode):
        def __enter__(self):
            _transform.EnterPassContext(self)
            return self

        def __exit__(self, ptype, value, trace):
            _transform.ExitPassContext(self)

        @staticmethod
        def current():
            """Return the current pass context."""
            return _transform.GetCurrentPassContext()

A ``PassContext`` object can be instantiated through the ``build_config`` API
which was used by Relay to configure the compilation options, including the
optimization level, fallback device for heterogeneous execution, and
required/disabled passes.

Pass Objects
^^^^^^^^^^^^

``Pass`` is the base class of all pass objects. All methods here are just simple
wrappers that were implemented in the backend. They are defined for users to
conveniently interact with the base class in Python. Only a ``__call__`` is
defined in the pass base class to make the subclasses as callable objects so
that they can be invoked easily (e.g., ``pass_xx(arg)``) for execution.

.. code:: python

    @register_relay_node
    class Pass(RelayNode):
       def __call__(self, mod):
           return _transform.RunPass(self, mod)

Some auxiliary APIs are provided to enable easy creation of passes from
the Python frontend and to let the pass infra control the execution. For
example, ``module_pass``, ``function_pass``, and ``sequential`` are provided to
users so that they can customize their own pass or pass pipeline.

For all the passes that are implemented in the C++ backend, we provide
a corresponding Python API in `python/tvm/relay/transform.py`_. For instance,
const folding has a Python API like the following:

.. code:: python

    def FoldConstant():
        return _transform.FoldConstant()

Users can build a pass through decoration like the following:

.. code:: python

    @relay.transform.module_pass(opt_level=2)
    def transform(mod, ctx):
       tp = relay.TensorType((10,), "float32")
       x = relay.var("x", tp)
       gv = relay.GlobalVar("abs")
       func = relay.Function([x], relay.abs(x))
       new_mod = relay.Module({gv: func})
       new_mod.update(mod)
       return new_mod

   module_pass = transform
   assert isinstance(module_pass, transform.ModulePass)
   assert module_pass.info.opt_level == 2

The ``transform`` function here adds an ``abs`` function to the input module,
but it could be any customized optimizations at the module level. After
creating this ``module_pass``, users can apply it on any Relay module. For
example, we can build an empty module and apply this pass to add an ``abs``
function.

.. code:: python

    mod = relay.Module()
    mod = module_pass(mod)

Correspondingly, we also offer such functionality for ``function_pass``. For
instance, an example function-level pass could be written as the following:

.. code:: python

    @relay.transform.function_pass(opt_level=1)
    class TestReplaceFunc:
       def __init__(self, new_func):
          self.new_func = new_func
          def transform_function(self, func, mod, ctx):
             # Just for demo purposes
             # Transform func to new_func
             return self.new_func

    x = relay.var("x", shape=(10, 20))
    f1 = relay.Function([x], x)
    f2 = relay.Function([x], relay.log(x))
    # fpass is now a special pass that replaces every
    # function to f1
    fpass = TestReplaceFunc(f1)
    # Now every function in input_mod is replaced by f1
    res_mod = fpass(input_mod)


Alternatively, users can also directly register a pass without using the
decorators and then invoke it. Let's use ``Sequential`` to demo this scenario.

Python Sequential Example
^^^^^^^^^^^^^^^^^^^^^^^^^

This example not only illustrates how users can directly create a sequential
pass using Python APIs (this could be applied to module- and function-level
passes as well), but also explains how we can build an optimization pipeline
using ``Sequential`` associated with other types of passes.

.. code:: python

    # Create a simple Relay program.
    shape = (1, 2, 3)
    c_data = np.array(shape).astype("float32")
    tp = relay.TensorType(shape, "float32")
    c = relay.const(c_data)
    x = relay.var("x", tp)
    y = relay.add(c, c)
    y = relay.multiply(y, relay.const(2, "float32"))
    y = relay.add(x, y)
    z = relay.add(y, c)
    z1 = relay.add(y, c)
    z2 = relay.add(z, z1)
    func = relay.Function([x], z2)

    # Customize the optimization pipeline.
    seq = tvm.transform.Sequential([
        relay.transform.InferType(),
        relay.transform.FoldConstant(),
        relay.transform.EliminateCommonSubexpr(),
        relay.transform.AlterOpLayout()
    ])

    # Create a module to perform optimizations.
    mod = relay.Module({"main": func})

    # Users can disable any passes that they don't want to execute by providing
    # a list, e.g. disabled_pass=["EliminateCommonSubexpr"].
    with relay.build_config(opt_level=3):
        with tvm.target.create("llvm"):
            # Perform the optimizations.
            mod = seq(mod)

Debugging
~~~~~~~~~

The pass infra provides a special pass (``PrintIR``) to dump the IR of the
whole module after applying a certain pass. A slightly modified version of the
sequential pass example could be like the following to enable IR dumping for
``FoldConstant`` optimization.

.. code:: python

    seq = tvm.transform.Sequential([
        relay.transform.InferType(),
        relay.transform.FoldConstant(),
        transform.PrintIR(),
        relay.transform.EliminateCommonSubexpr(),
        relay.transform.AlterOpLayout()
    ])

By inserting the ``PrintIR`` pass after ``FoldConstant``, the pass infra will
dump out the module IR when ``FoldConstant`` is done. Users can plug in this
pass after any pass they want to debug for viewing the optimization effect.

There is a more flexible debugging mechanism also exposed by the build configuration
object. One can pass a tracing function which can be used to execute arbitrary code
before and/or after each pass. A tracing function will receive a ``IRModule``, ``PassInfo``,
and a boolean indicating whether you are executing before, or after a pass.
An example is below.

.. code:: python

    def print_ir(mod, info, is_before):
        """Print the name of the pass, the IR, only before passes execute."""
        if is_before:
            print(f"Running pass: {}", info)
            print(mod)

    with relay.build_config(opt_level=3, trace=print_ir):
            with tvm.target.create("llvm"):
                # Perform the optimizations.
                mod = seq(mod)


For more pass infra related examples in Python and C++, please refer to
`tests/python/relay/test_pass_manager.py`_ and
`tests/cpp/relay_transform_sequential.cc`_, respectively.

.. _Sequential: https://pytorch.org/docs/stable/nn.html?highlight=sequential#torch.nn.Sequential

.. _Block: https://mxnet.incubator.apache.org/api/python/docs/api/gluon/block.html#gluon-block

.. _include/tvm/ir/transform.h: https://github.com/apache/incubator-tvm/blob/master/include/tvm/ir/transform.h

.. _src/relay/ir/transform.cc: https://github.com/apache/incubator-tvm/blob/master/src/relay/ir/transform.cc

.. _src/ir/transform.cc: https://github.com/apache/incubator-tvm/blob/master/src/ir/transform.cc

.. _src/relay/pass/fold_constant.cc: https://github.com/apache/incubator-tvm/blob/master/src/relay/pass/fold_constant.cc

.. _python/tvm/relay/transform.py: https://github.com/apache/incubator-tvm/blob/master/python/tvm/relay/transform.py

.. _tests/python/relay/test_pass_manager.py: https://github.com/apache/incubator-tvm/blob/master/tests/python/relay/test_pass_manager.py

.. _tests/cpp/relay_transform_sequential.cc: https://github.com/apache/incubator-tvm/blob/master/tests/cpp/relay_transform_sequential.cc

.. _include/tvm/relay/transform.h: https://github.com/apache/incubator-tvm/blob/master/include/tvm/relay/transform.h
