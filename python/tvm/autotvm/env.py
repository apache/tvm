"""Global configuration/variable scope for autotvm"""

class AutotvmGlobalScope(object):
    current = None

    def __init__(self):
        self._old = AutotvmGlobalScope.current
        AutotvmGlobalScope.current = self

        self.cuda_target_arch = None

global_scope = AutotvmGlobalScope()
