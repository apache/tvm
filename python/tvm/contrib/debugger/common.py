"""Common values and methods for TVM Debugger."""

class UxAction(object):
    """Enum-like values for possible action to take on start of a run() call."""

    # Run once with debug tensor-watching.
    DEBUG_RUN = "debug_run"

    # Run without debug tensor-watching.
    NON_DEBUG_RUN = "non_debug_run"

    # Finished run, can exit.
    EXIT = "exit"
