"""Methods and data structures to support dumping HalideIR to Hybrid Script.
This allows users to do quick hack to generated HalideIR and cast it back to
TVM modules.
"""

from .. import build_module
from .. import ir_pass
from .. import schedule
from .. import stmt as _stmt
from .. import expr as _expr

class HybridModule(object):
    """The usage of Hybrid Module is very similar to conventional TVM module,
    but conventional TVM module requires a function body which is already fully
    lowered. This contradicts to the fact that Hybrid Module is originally a text
    format for Phase 0 HalideIR. Thus, a totally separated module is defined."""

    def __init__(self, src):
        self.src_ = src

    def __call__(self):
        pass

    def get_source(self):
        return self.src_
