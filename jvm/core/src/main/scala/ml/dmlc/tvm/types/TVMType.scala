package ml.dmlc.tvm.types

object TVMType {
  private val CODE2STR = Map(
    0 -> "int",
    1 -> "uint",
    2 -> "float",
    4 -> "handle"
  )
}

class TVMType(typeStr: String, val lanes: Int = 1) {
  private val (typeCodeTemp, bitsTemp) =
    if (typeStr.startsWith("int")) {
      (0, typeStr.substring(3).toInt)
    } else if (typeStr.startsWith("uint")) {
      (1, typeStr.substring(4).toInt)
    } else if (typeStr.startsWith("float")) {
      (2, typeStr.substring(5).toInt)
    } else if (typeStr.startsWith("handle")) {
      (4, 64)
    } else {
      throw new IllegalArgumentException("Do not know how to handle type " + typeStr)
    }

  val typeCode = typeCodeTemp
  val bits = if (bitsTemp == 0) 32 else bitsTemp
  if ((bits & (bits - 1)) != 0 || bits < 8) {
    throw new IllegalArgumentException("Do not know how to handle type " + typeStr)
  }

  def numOfBytes: Int = {
    bits / 8
  }

  override def hashCode: Int = {
    (typeCode << 16) | (bits  << 8) | lanes
  }

  override def equals(other: Any): Boolean = {
    if (other != null && other.isInstanceOf[TVMType]) {
      val otherInst = other.asInstanceOf[TVMType]
      (bits == otherInst.bits) && (typeCode == otherInst.typeCode) && (lanes == otherInst.lanes)
    } else {
      false
    }
  }

  override def toString: String = {
    val str = s"${TVMType.CODE2STR(typeCode)}$bits"
    if (lanes != 1) {
      str + lanes
    } else {
      str
    }
  }
}
