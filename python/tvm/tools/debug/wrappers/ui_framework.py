"""Framework of debug wrapper sessions."""
from __future__ import absolute_import

import abc

from tvm.tools.debug.util import common

# Helper function.
def _check_type(obj, expected_types):
    """Check if an object is of the expected type.

    Args:
      obj: The object being checked.
      expected_types: (`type` or an iterable of `type`s) The expected `type`(s)
        of obj.

    Raises:
        TypeError: If obj is not an instance of expected_type.
    """
    if not isinstance(obj, expected_types):
        raise TypeError("Expected type %s; got type %s" %
                        (expected_types, type(obj)))


class OnSessionInitRequest(object):
    """Request to an on-session-init callback.

    This callback is invoked during the __init__ call to a debug-wrapper session.
    """

    def __init__(self, sess):
        """Constructor.

        Args:
          sess: A TVM Session object.
        """

        self.session = sess


class OnSessionInitAction(object):
    """Enum-like values for possible action to take on session init."""

    # Proceed, without special actions, in the wrapper session initialization.
    # What action the wrapper session performs next is determined by the caller
    # of the wrapper session. E.g., it can call run().
    PROCEED = "proceed"

    # Instead of letting the caller of the wrapper session determine what actions
    # the wrapper session will perform next, enter a loop to receive instructions
    # from a remote client.
    # For example, TensorBoard visual debugger can use this action so that it can
    # launch session.run() calls remotely.
    REMOTE_INSTR_LOOP = "remote_instr_loop"


class OnSessionInitResponse(object):
    """Response from an on-session-init callback."""

    def __init__(self, action):
        """Constructor.

        Args:
          action: (`OnSessionInitAction`) Debugger action to take on session init.
        """
        _check_type(action, str)
        self.action = action


class OnRunStartRequest(object):
    """Request to an on-run-start callback.

    This callback is invoked during a run() call of the debug-wrapper
    session, immediately after the run() call counter is incremented.
    """

    def __init__(self, outputs, input_dict, run_options, run_metadata,
                 run_call_count, is_callable_runner=False):
        """Constructor of `OnRunStartRequest`.

        Args:
          outputs: Output targets of the run() call.
          input_dict: The input dictionary to the run() call.
          run_options: RunOptions input to the run() call.
          run_metadata: RunMetadata input to the run() call.
            The above four arguments are identical to the input arguments to the
            run() method of a non-wrapped TVM session.
          run_call_count: 1-based count of how many run calls (including this one)
            has been invoked.
          is_callable_runner: (bool) whether a runner returned by
            Session.make_callable is being run.
        """
        self.outputs = outputs
        self.input_dict = input_dict
        self.run_options = run_options
        self.run_metadata = run_metadata
        self.run_call_count = run_call_count
        self.is_callable_runner = is_callable_runner


class OnRunStartResponse(object):
    """Request from an on-run-start callback.

    The caller of the callback can use this response object to specify what
    action the debug-wrapper session actually takes on the run() call.
    """

    def __init__(self,
                 action,
                 debug_urls,
                 debug_ops="DebugIdentity",
                 node_name_regex_whitelist=None,
                 op_type_regex_whitelist=None,
                 tensor_dtype_regex_whitelist=None,
                 tolerate_dbg_op_failures=False):
        """Constructor of `OnRunStartResponse`.

        Args:
          action: (`CLIRunStartAction`) the action actually taken by the wrapped
            session for the run() call.
          debug_urls: (`list` of `str`) debug_urls used in watching the tensors
            during the run() call.
          debug_ops: (`str` or `list` of `str`) Debug op(s) to be used by the
            debugger.
          node_name_regex_whitelist: Regular-expression whitelist for node
            name.
          op_type_regex_whitelist: Regular-expression whitelist for op type.
          tensor_dtype_regex_whitelist: Regular-expression whitelist for tensor
            dtype.
          tolerate_dbg_op_failures: Whether debug op creation failures
            are to be tolerated.
        """

        _check_type(action, str)
        self.action = action

        _check_type(debug_urls, list)
        self.debug_urls = debug_urls

        self.debug_ops = debug_ops

        self.node_name_regex_whitelist = node_name_regex_whitelist
        self.op_type_regex_whitelist = op_type_regex_whitelist
        self.tensor_dtype_regex_whitelist = tensor_dtype_regex_whitelist
        self.tolerate_dbg_op_failures = (tolerate_dbg_op_failures)

class OnRunEndRequest(object):
    """Request to an on-run-end callback.

    The callback is invoked immediately before the wrapped run() call ends.
    """

    def __init__(self,
                 performed_action,
                 run_metadata=None,
                 tvm_error=None):
        """Constructor for `OnRunEndRequest`.

        Args:
          performed_action: (`CLIRunStartAction`) Actually-performed action by the
            debug-wrapper session.
          run_metadata: run_metadata output from the run() call (if any).
          tvm_error: (errors.OpError subtypes) TVM OpError that occurred
            during the run (if any).
        """

        _check_type(performed_action, str)
        self.performed_action = performed_action

        self.run_metadata = run_metadata
        self.tvm_error = tvm_error


class OnRunEndResponse(object):
    """Response from an on-run-end callback."""

    def __init__(self):
        # Currently only a placeholder.
        pass


class CLIRunCommand(object):
    """Run command created based on the CLI user input.

    Contain the CLI user input and other parameters to invoke graph runtime run
    and trigger back the CLI on run end to handle the output.
    """

    def __init__(self, run_start_resp, metadata):
        """Constructor of CLIRunCommand.

        Args:
          run_start_resp: Run start is depend on the action triggered from the CLI.
          RUN output also depend on the action saved in 'run_start_resp'.
          metadata: Same as meta data argument
        """
        self._run_metadata = metadata
        self._run_start_resp = run_start_resp

    def get_run_metadata(self):
        return self._run_metadata

    def get_run_start_resp(self):
        return self._run_start_resp


class BaseDebugWrapperModule(object):
    """Base class of debug-wrapper session classes.

    Concrete classes that inherit from this class need to implement the abstract
    methods such as on_session_init, on_run_start and on_run_end.
    """

    def __init__(self, sess, graph, ctx=None, thread_name_filter=None,
                 pass_through_operrors=False):
        """Constructor of `BaseDebugWrapperModule`.

        Args:
          sess: An (unwrapped) TVM session instance. It should be a subtype
            of `BaseSession` or `tf.MonitoredSession`.
          thread_name_filter: Regular-expression filter (whitelist) for name(s) of
            thread(s) on which the wrapper session will be active. This regular
            expression is used in a start-anchored fashion on the thread name, i.e.,
            by applying the `match` method of the compiled pattern. The default
            `None` means that the wrapper session will be active on all threads.
            E.g., r"MainThread$", r"QueueRunnerThread.*".
          pass_through_operrors: If True, all captured OpErrors will be
            propagated.  By default this captures all OpErrors.

        Raises:
          ValueError: On invalid `OnSessionInitAction` value.
          NotImplementedError: If a non-DirectSession sess object is received.
        """
        self._outputs = []
        self._input_dict = {}
        self._graph = graph
        self._ctx = ctx

        # The session being wrapped.
        self._sess = sess
        self._thread_name_filter_pattern = (re.compile(thread_name_filter)
                                            if thread_name_filter else None)
        self._pass_through_operrors = pass_through_operrors

        # Keeps track of number of run calls that have been performed on this
        # debug-wrapper session. The count can be used for purposes such as
        # displaying the state of the Session in a UI and determining a run
        # number-dependent debug URL.
        self._run_call_count = 0

        # Invoke on-session-init callback.
        response = self.on_session_init(OnSessionInitRequest(self._sess))
        _check_type(response, OnSessionInitResponse)

        if response.action == OnSessionInitAction.PROCEED:
            pass
        elif response.action == OnSessionInitAction.REMOTE_INSTR_LOOP:
            raise NotImplementedError(
                "OnSessionInitAction REMOTE_INSTR_LOOP has not been "
                "implemented.")
        else:
            raise ValueError(
                "Invalid OnSessionInitAction value: %s" % response.action)

    def set_ouputs(self, name):
        """Set the output Name which used to access from runtime.

        Args:
          name : Name of the output by which used to output from runtime.
        """
        self._outputs.append(name)

    def set_input(self, name, value):
        """Set the input with Name and Numpy/Tvm.NdArray Value.

        Args:
          name : Name of the input by which used to input to runtime.
          value : Numpy/Tvm.NdArray instance which used to input to runtime.
        """
        self._input_dict.update({name: value})

    def dump_folder(self, folder_name=None):
        """Sets and returns the folder to dump the outputs and graph.

        Args:
          folder_name : String the name of folder

        """
        if folder_name:
            self._dump_folder = folder_name
        return self._dump_folder

    def run_end(self, cli_command, retvals):
        """Notify CLI that the graph runtime is completed the task and output
        is ready access from CLI.

        Args:
          cli_command: CLI command is created by the CLI wrapper before invoking
          graph runtime.
          retvals: graph runtime return value.

        Returns:
          None
        """
        retvals = retvals
        run_start_resp = cli_command.get_run_start_resp()
        if run_start_resp.action == common.CLIRunStartAction.DEBUG_RUN:
            run_metadata = cli_command.get_run_metadata()
            run_end_req = OnRunEndRequest(
                run_start_resp.action,
                run_metadata=run_metadata)
            # Invoke on-run-end callback and obtain response.
            run_end_resp = self.on_run_end(run_end_req)
            _check_type(run_end_resp, OnRunEndResponse)
            # Currently run_end_resp is only a placeholder. No action is taken on it.
        elif run_start_resp.action == common.CLIRunStartAction.NON_DEBUG_RUN:
            run_end_req = OnRunEndRequest(run_start_resp.action)

    def get_run_command(self,
                        outputs=None,
                        options=None,
                        run_metadata=None,
                        callable_runner=None):
        """Wrapper around Session.run() that inserts tensor watch options.

        Args:
          outputs: Same as the `outputs` arg to regular `Session.run()`.
          options: Same as the `options` arg to regular `Session.run()`.
          run_metadata: Same as the `run_metadata` arg to regular `Session.run()`.
          callable_runner: A `callable` returned by `Session.make_callable()`.
            If not `None`, `outputs` and `input_dict` must both be `None`.

        Returns:
          Simply forwards the output of the wrapped `Session.run()` call.

        Raises:
          ValueError: On invalid `CLIRunStartAction` value. Or if `callable_runner`
            is not `None` and either or both of `outputs` and `input_dict` is `None`.
        """
        retvals = True
        outputs = self._outputs

        if not callable_runner:
            self.increment_run_call_count()
        else:
            if outputs or self._input_dict:
                raise ValueError(
                    "callable_runner and outputs/input_dict are mutually exclusive, but "
                    "are used simultaneously.")

        # Invoke on-run-start callback and obtain response.
        run_start_resp = self.on_run_start(
            OnRunStartRequest(self._outputs, self._input_dict, options, run_metadata,
                              self._run_call_count,
                              is_callable_runner=bool(callable_runner)))
        _check_type(run_start_resp, OnRunStartResponse)

        run_command = CLIRunCommand(run_start_resp, run_metadata)

        if run_start_resp.action == common.CLIRunStartAction.DEBUG_RUN or \
               run_start_resp.action == common.CLIRunStartAction.NON_DEBUG_RUN:
            return run_command
        else:
            raise ValueError(
                "Invalid CLIRunStartAction value: %s" % run_start_resp.action)
        return retvals

    @property
    def run_call_count(self):
        """Get the number how many time call run is invoked.

        Returns:
          ('int') number of run count.
        """
        return self._run_call_count

    def increment_run_call_count(self):
        """Increment the run invoke counter."""
        self._run_call_count += 1

    @abc.abstractmethod
    def on_session_init(self, request):
        """Callback invoked during construction of the debug-wrapper session.

        This is a blocking callback.
        The invocation happens right before the constructor ends.

        Args:
          request: (`OnSessionInitRequest`) callback request carrying information
            such as the session being wrapped.

        Returns:
          An instance of `OnSessionInitResponse`.
        """

    @abc.abstractmethod
    def on_run_start(self, request):
        """Callback invoked on run() calls to the debug-wrapper session.

        This is a blocking callback.
        The invocation happens after the wrapper's run() call is entered,
        after an increment of run call counter.

        Args:
          request: (`OnRunStartRequest`) callback request object carrying
            information about the run call such as the outputs, input dict, run
            options, run metadata, and how many `run()` calls to this wrapper
            session have occurred.

        Returns:
          An instance of `OnRunStartResponse`, carrying information to
            1) direct the wrapper session to perform a specified action (e.g., run
              with or without debug tensor watching.)
            2) debug URLs used to watch the tensors.
        """

    @abc.abstractmethod
    def on_run_end(self, request):
        """Callback invoked on run() calls to the debug-wrapper session.

        This is a blocking callback.
        The invocation happens right before the wrapper exits its run() call.

        Args:
          request: (`OnRunEndRequest`) callback request object carrying information
            such as the actual action performed by the session wrapper for the
            run() call.

        Returns:
          An instance of `OnRunStartResponse`.
        """
