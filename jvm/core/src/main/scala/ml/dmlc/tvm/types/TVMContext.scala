package ml.dmlc.tvm.types

// TVM context structure
object TVMContext {
  val MASK2STR = Map(
    1 -> "cpu",
    2 -> "gpu",
    4 -> "opencl",
    8 -> "metal",
    9 -> "vpi"
  )
  val STR2MASK = Map(
    "cpu" -> 1,
    "gpu" -> 2,
    "cuda" -> 2,
    "cl" -> 4,
    "opencl" -> 4,
    "metal" -> 8,
    "vpi" -> 9
  )
}

class TVMContext(val deviceType: String, val deviceId: Int) {
  // Whether this device exist.
  def exist: Boolean = {
    //_api_internal._GetDeviceAttr(
    //  self.device_type, self.device_id, 0) != 0
    ???
  }

/*
@property
def max_threads_per_block (self):
"""Maximum number of threads on each block."""
return _api_internal._GetDeviceAttr (
self.device_type, self.device_id, 1)

@property
def warp_size (self):
"""Number of threads that executes in concurrent."""
return _api_internal._GetDeviceAttr (
self.device_type, self.device_id, 2)

def sync (self):
"""Synchronize until jobs finished at the context."""
check_call (_LIB.TVMSynchronize (self, None) )

def __eq__ (self, other):
return (isinstance (other, TVMContext) and
self.device_id == other.device_id and
self.device_type == other.device_type)

def __ne__ (self, other):
return not self.__eq__ (other)

def __repr__ (self):
if self.device_type >= RPC_SESS_MASK:
tbl_id = self.device_type / RPC_SESS_MASK - 1
dev_type = self.device_type % RPC_SESS_MASK
return "remote[%d]:%s(%d)" % (
tbl_id, TVMContext.MASK2STR[dev_type], self.device_id)
return "%s(%d)" % (
TVMContext.MASK2STR[self.device_type], self.device_id)
*/
}
