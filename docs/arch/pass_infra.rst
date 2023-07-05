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

.. _pass-infra:

Pass Infrastructure
===================

Both Relay and TVM IR contain a series of optimization passes which improve performance metrics
of models such as mean inference, memory footprint, or power consumption for
specific devices. There is a suite of standard optimizations as well as machine
learning-specific optimizations including constant folding, dead code
elimination, operator layout alteration, operator fusion, buffer handling, and
loop transformation, etc. Each of these passes is structured as a ir-to-ir
transformation using the analysis result collected during and/or before traversal.

However, as TVM evolves quickly, the need for a more systematic and efficient
way to manage these passes is becoming apparent. In addition, a generic
framework that manages the passes across different layers of the TVM stack (e.g.
Relay and tir) paves the way for developers to quickly prototype and plug the
implemented passes into the system.

This doc describes the design of such an infra that takes the advantage of the
way production compilers are used to manage the optimization passes and the style
modern deep learning frameworks adopted to build up layers.

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

The design of the Relay pass infra is largely inspired by the hierarchical
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

    class PassInfoNode : public Object {
      String name;
      int opt_level;
      Array<String> required;
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
at ``opt_level=3`` and exclude those in the disabled pass list. ``PassContext``
also provides a way to instrument all passes. See section :ref:`pass_instrument_cpp_backend`.

This class is designed for users to conveniently write the Python ``with``
syntax to perform optimizations under a certain configuration. In addition, the
users can obtain the context that is available within a certain program scope in
a thread-safe way through ``PassContext::Current()``, since a thread-local store
``PassContextThreadLocalStore`` is used to hold the created pass context
objects. Examples will be provided later to show how we can use both the C++ and
Python APIs to create a compilation pipeline using pass context.

.. code:: c++

    class PassContextNode : public Object {
     public:
      int opt_level{2};
      tvm::Array<tvm::Expr> required_pass;
      tvm::Array<tvm::Expr> disabled_pass;
      mutable Optional<DiagnosticContext> diag_ctx;
      Map<String, ObjectRef> config;
      Array<instrument::PassInstrument> instruments;
    };

    class PassContext : public NodeRef {
     public:
      TVM_DLL static PassContext Create();
      TVM_DLL static PassContext Current();
      TVM_DLL void InstrumentEnterPassContext();
      TVM_DLL void InstrumentExitPassContext();
      TVM_DLL bool InstrumentBeforePass(const IRModule& mod, const PassInfo& info) const;
      TVM_DLL void InstrumentAfterPass(const IRModule& mod, const PassInfo& info) const;
      /* Other fields are omitted. */

     private:
      // The entry of a pass context scope.
      TVM_DLL void EnterWithScope();
      // The exit of a pass context scope.
      TVM_DLL void ExitWithScope();

      // Classes to get the Python `with` like syntax.
      friend class tvm::With<PassContext>;
    };

    struct PassContextThreadLocalEntry {
      /*! \brief The default pass context. */
      PassContext default_context;
      /*! \brief The current pass context. */
      std::stack<PassContext> context_stack;
      PassContextThreadLocalEntry() {
        default_context = PassContext(make_node<PassContextNode>());
      }
    };

    /*! \brief The thread-local store to hold the pass context. */
    typedef dmlc::ThreadLocalStore<PassContextThreadLocalEntry>
         PassContextThreadLocalStore;

Pass Constructs
^^^^^^^^^^^^^^^

The pass infra is designed in a hierarchical manner, and it could work at
different granularities of Relay/tir programs. A pure virtual class ``PassNode`` is
introduced to serve as the base of the different optimization passes. This class
contains several virtual methods that must be implemented by the
subclasses at the level of modules, functions, or sequences of passes.

.. code:: c++

    class PassNode : Object {
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
level, users can even add and/or delete functions in a module. Note that all
passes

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
optimizations for a given Relay/tir module. It fetches one function at a time from
the function list of a module for optimization and yields a rewritten Relay
``Function`` or tir ``PrimFunc``. Most of passes can be classified into this category, such as
common subexpression elimination and inference simplification in Relay as well as vectorization
and flattening storage in tir, etc.

Note that the scope of passes at this level is either a Relay function or a tir primitive function.
Therefore, we cannot add or delete a function through these passes as they are not aware of
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
        ICHECK(pass.defined()) << "Found undefined pass for optimization.";
        const PassInfo& pass_info = pass->Info();
        if (!PassEnabled(pass_info))  continue;
        for (const auto& it : pass_info->required) {
          const auto* name = it.as<tvm::ir::StringImm>();
          ICHECK(name);
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
      ICHECK(f != nullptr) << "Cannot find " << fpass_name
                          << "to create the pass " << pass_name;
      return (*f)();
    }

Some helper functions are provided to create each type of these aforementioned
passes. These helpers are also exposed to the Python frontend for users to
favorably use Python APIs to create a specific pass object.

.. code:: c++

    Pass CreateFunctionPass(
        const runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)>& pass_func,
        int opt_level,
        String name,
        Array<String> required);

    Pass CreatePrimFuncPass(
        const runtime::TypedPackedFunc<PrimFunc(PrimFunc, IRModule, PassContext)>& pass_func,
        int opt_level,
        String name,
        Array<String> required);

    Pass CreateModulePass(
        const runtime::TypedPackedFunc<IRModule(IRModule, PassContext)>& pass_func,
        int opt_level,
        String name,
        Array<String> required);

    Pass Sequential(tvm::Array<Pass> passes, PassInfo pass_info);

Pass Registration
^^^^^^^^^^^^^^^^^

We've covered the concept of different level of passes and the context used for
compilation. It would be interesting to see how easily users can register
a pass.  Let's take const folding as an example. This pass has already been
implemented to fold constants in a Relay function (found in
`src/relay/transforms/fold_constant.cc`_).

An API was provided to perform the ``Expr`` to ``Expr`` transformation.

.. code:: c++

    Expr FoldConstant(const Expr& expr);

In order to register this pass to the pass infra, we first need to decide at
which level this pass will be performed. As const folding happens on individual
functions, we should intuitively create a ``FunctionPass`` for it through
``CreateFunctionPass``. The ``pass_func`` is returned as a packed function that
invokes the ``Expr`` to ``Expr`` API on each function in a `IRModule`. ``{}``
indicates that no prerequisite is required for this pass. Otherwise, the pass
developer has to identify and list them.

Meanwhile, a pass API endpoint is registered with the name
``relay._transform.FoldConstant``. This pass, therefore, becomes an entry in the
registry that can be accessed by both C++ (e.g. the ``GetPass`` above) and
Python when needed.

.. code:: c++

    namespace transform {

    Pass FoldConstant() {
      runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
        [=](Function f, IRModule m, PassContext pc) {
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

.. _pass_instrument_cpp_backend:

Pass Instrument
^^^^^^^^^^^^^^^

Pass Instrument is a mechanism to analyze the pass itself. For example,
we can use the infrastructure to know how much time and memory a pass requires
or how a pass can transform the IR module.

We introduce four instrument points in the life-cycle of ``PassContext``.

.. code:: c++

    TVM_DLL void InstrumentEnterPassContext();
    TVM_DLL void InstrumentExitPassContext();
    TVM_DLL bool InstrumentBeforePass(const IRModule& mod, const PassInfo& info) const;
    TVM_DLL void InstrumentAfterPass(const IRModule& mod, const PassInfo& info) const;

``InstrumentEnterPassContext`` is called immediately when entering the scope
of the ``PassContext`` instance.

``InstrumentExitPassContext`` is called when leaving the scope of ``PassContext``,
or exceptions occur during the execution of passes.
This method is also called when instruments is being overriden by ``override_instruments`` in :py:class:`tvm.transform.PassContext`.
See :ref:`pass_instrument_overriden`.

``InstrumentBeforePass`` is called before execution.
``InstrumentAfterPass`` is called after execution if the pass should be run. The behavior is like:

.. code:: c++

      if (pass_ctx.InstrumentBeforePass(ir_module, pass_info)) {
        new_ir_module = run_pass(ir_module, pass_ctx);
        pass_ctx.InstrumentAfterPass(new_ir_module, pass_info);
        return new_ir_module;
      }

The ``PassInstrument`` interface allow you to run arbitrary code inside above four methods.
Multiple ``PassInstrument`` instances can be registed into a single
``PassContext``. ``PassInstrument`` instances are called sequentially in the order of
``instruments`` argument passed to ``PassContext``.

``PassInstrument`` provides following interfaces:

.. code:: c++

    namespace instrument {

    class PassInstrumentNode : public Object {
     public:
      String name;
      virtual void EnterPassContext() const = 0;
      virtual void ExitPassContext() const = 0;
      virtual bool ShouldRun(const IRModule& mod, const transform::PassInfo& info) const = 0;
      virtual void RunBeforePass(const IRModule& mod, const transform::PassInfo& info) const = 0;
      virtual void RunAfterPass(const IRModule& mod, const transform::PassInfo& info) const = 0;
      /* Other fields are omitted. */
    };

    class PassInstrument : public ObjectRef {
     public:
      TVM_DEFINE_OBJECT_REF_METHODS(PassInstrument, ObjectRef, PassInstrumentNode);
    };

    }  // namespace instrument

Python frontend are provided to implement ``PassInstrument`` quickly. See :ref:`pass_instrument_py_frontend`.

Within a ``PassContext``, the call sequence of a ``PassInstrument`` instance is like:

::

    with PassContext(instruments=[pi]) # pi = a PassInstrument implementation.
        pi.EnterPassContext()

        if pi.ShouldRun(Pass1):
            pi.RunBeforePass()
            Pass1()
            pi.RunAfterPass()

        if pi.ShouldRun(Pass2):
            pi.RunBeforePass()
            Pass2()
            pi.RunAfterPass()

        pi.ExitPassContext()

Here is a brief introduction of relations between ``PassInstrument`` interfaces
and ``PassContext`` methods. See (`src/ir/transform.cc`_) for more details.

- ``InstrumentEnterPassContext``

  * ``EnterPassContext()`` is executed in the order of ``instruments`` passed to the ``PassContext``.
  * When an exception raises, ``PassContext`` disable the pass instrumentation
    by clearing all registered ``PassInstrument`` instances.
  * Then ``PassContext`` execute ``ExitPassContext()`` method of each ``PassInstrument``
    instances which successfully finished ``EnterPassContext()``
  * For example, if ``PassInstrument`` A, B, and C are registered to a ``PassContext``
    and A finished ``EnterPassContext()`` while B throws an exception, then C
    is never executed; ``ExitPassContext()`` of A is executed.

- ``InstrumentExitPassContext``

  * ``ExitPassContext()`` of each ``PassInstrument`` instances are executed in
    the order of ``instruments`` passed to the ``PassContext``.
  * While an exception occurs, ``instruments`` is cleared.
  * ``PassInstrument`` Instances registered after the one throwing exceptions do not execute ``ExitPassContext``.

- ``InstrumentBeforePass``

  * ``ShouldRun`` is executed if the pass is not listed as a required pass.
  * ``RunBeforePass`` is executed in the order of ``instruments`` if the pass is not blocked by ``ShouldRun``.
  * Note that ``InstrumentBeforePass`` returns a boolean indicating whether or not the pass should be run.
  * When an exception occur, it is thrown immediately.
    We rely on Python Context Manager to exit ``PassContext`` safely
    (meaning ``ExitPassContext`` of each instruments will be run. For C++, please refer to `include/tvm/support/with.h`_.)

- ``InstrumentAfterPass``

  * ``RunAfterPass`` is executed in the order of ``instruments`` passed to the ``PassContext``.
  * When an exception occur, it is thrown immediately.
    We rely on Python Context Manager or ``With`` class(`include/tvm/support/with.h`_) to exit ``PassContext`` safely

Built-in Instrument
^^^^^^^^^^^^^^^^^^^

There are several built-in instruments. Those marked with *TODO* are not implemented yet.

- PassTimingInstrument (see `src/ir/instrument.cc`_)

  * Profile the execution time of passes.

- PrintIRBefore(TODO)

  * Print the IR module before the pass transforms it. :py:func:`tvm.transform.PrintIR`
    can also serve this purpose if we insert it around passes. However,
    with the ``PassInstrument``, we don't need to modify the sequence of passes.

- PrintAfter(TODO)

  * Print the IR module after the pass transforms it.

Python Frontend
~~~~~~~~~~~~~~~

Only some simple APIs are needed for the frontend side. For example, we can
provide users the following APIs to create and execute a pass (full
implementation is provided in `python/tvm/relay/transform/transform.py`_ and
`python/tvm/ir/transform.py`_). The backend
receives the information and decides which function it should use to create
a Pass object.

PassContext
^^^^^^^^^^^

Python frontend provides a wrapper for the ``PassContext`` to enable the
``with`` syntax by overriding ``__enter__`` and ``__exit__``. A ``current``
static method is offered for users to get the context that is in use under
a certain scope.

.. code:: python

    @tvm._ffi.register_object("transform.PassContext")
    class PassContext(tvm.runtime.Object):
        def __enter__(self):
            _transform.EnterPassContext(self)
            return self

        def __exit__(self, ptype, value, trace, config):
            _transform.ExitPassContext(self)

        @staticmethod
        def current():
            """Return the current pass context."""
            return _transform.GetCurrentPassContext()

A ``PassContext`` is used to configure the compilation options, including the
optimization level and required/disabled passes. It can also take a dictionary
of configs so that different passes can conveniently fetch the passed data, such
as fallback device info and step/depth for loop unrolling, etc. In order to
enable fetching the required config, the key must be registered through
``TVM_REGISTER_PASS_CONFIG_OPTION``. For example, the following is used by the
loop unrolling pass

.. code:: c++

    TVM_REGISTER_PASS_CONFIG_OPTION("tir.UnrollLoop", UnrollLoopConfig);

Please refer to `src/tir/transforms/unroll_loop.cc`_ for more details.

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
corresponding Python APIs in `python/tvm/ir/transform.py`_ and
`python/tvm/relay/transform/transform.py`_, respectively. For instance,
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
       new_mod = tvm.IRModule({gv: func})
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

    mod = tvm.IRModule()
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
decorators and then invoke it. For more examples about how to customize your own
optimization pipeline and debug Relay and tir passes, please refer to the
`use pass infra`_ tutorial.


.. _pass_instrument_py_frontend:

Pass Instrument
^^^^^^^^^^^^^^^

One can implement a ``PassInstrument`` by using the ``pass_instrument``
decorator(`python/tvm/ir/instrument.py`_) on a class implementing following methods.
Note that it is recommended to use the ``pass_instrument`` decorator to implement
``PassInstrument``, instead of overriding or subclassing.

- ``enter_pass_ctx``

  * This method is run when entering ``PassContext``.

- ``exit_pass_ctx``

  * This method is run when exiting ``PassContext``.

- ``should_run``

  * This method is run before a pass is executed, returning a boolean
    indicating whether or not the pass should be run.

- ``run_before_pass``

  * If a pass should be run, this method is run just before pass execution.

- ``run_after_pass``

  * This method is run right after a pass has been executed.

``PassInstrument`` instances can be registered through ``instruments`` argument in
:py:class:`tvm.transform.PassContext`.

`use pass instrument`_ tutorial provides examples for how to implement ``PassInstrument`` with Python APIs.

.. _pass_instrument_overriden:

Override Instruments in Current PassContext
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``override_instruments`` method is provided to override the ``instruments`` of current ``PassContext``.
For example, if passes are run without explicitly creating a new ``PassContext``,
one can still register ``PassInstrument`` into the global ``PassContext`` by:

.. code:: python

    cur_pass_ctx = tvm.transform.PassContext.current()
    # override PassInstrument instances
    cur_pass_ctx.override_instruments([pass_inst])
    mod = pass_seq(mod)
    result = pass_inst.get_result()

Note that when ``override_instruments`` is called, the ``exit_pass_ctx`` method of
old ``PassInstrument`` instances are called. Then the ``enter_pass_ctx`` method of
new ``PassInstrument`` are called.

.. _Sequential: https://pytorch.org/docs/stable/nn.html?highlight=sequential#torch.nn.Sequential

.. _Block: https://mxnet.apache.org/api/python/docs/api/gluon/block.html#gluon-block

.. _include/tvm/ir/transform.h: https://github.com/apache/tvm/blob/main/include/tvm/ir/transform.h

.. _include/tvm/support/with.h: https://github.com/apache/tvm/blob/main/include/tvm/support/with.h

.. _src/relay/ir/transform.cc: https://github.com/apache/tvm/blob/main/src/relay/ir/transform.cc

.. _src/ir/transform.cc: https://github.com/apache/tvm/blob/main/src/ir/transform.cc

.. _src/ir/instrument.cc: https://github.com/apache/tvm/blob/main/src/ir/instrument.cc

.. _src/relay/transforms/fold_constant.cc: https://github.com/apache/tvm/blob/main/src/relay/transforms/fold_constant.cc

.. _python/tvm/relay/transform/transform.py: https://github.com/apache/tvm/blob/main/python/tvm/relay/transform/transform.py

.. _include/tvm/relay/transform.h: https://github.com/apache/tvm/blob/main/include/tvm/relay/transform.h

.. _python/tvm/ir/transform.py: https://github.com/apache/tvm/blob/main/python/tvm/ir/transform.py

.. _python/tvm/ir/instrument.py: https://github.com/apache/tvm/blob/main/python/tvm/ir/instrument.py

.. _src/tir/transforms/unroll_loop.cc: https://github.com/apache/tvm/blob/main/src/tir/transforms/unroll_loop.cc

.. _use pass infra: https://github.com/apache/tvm/blob/main/tutorials/dev/use_pass_infra.py

.. _use pass instrument: https://github.com/apache/tvm/blob/main/tutorials/dev/use_pass_instrument.py
