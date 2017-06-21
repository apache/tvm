package ml.dmlc.tvm

object API {
  private val function = Function.initAPI(
    name => name.indexOf('.') == -1 && !name.startsWith("_"),
    name => name)

  def apply(name: String): Function = function(name)
}

object APIInternal {
  private val function = Function.initAPI(
    name => name.indexOf('.') == -1 && name.startsWith("_"),
    name => name)

  def apply(name: String): Function = function(name)
}
