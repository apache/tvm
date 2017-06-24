/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ml.dmlc.tvm

/**
 * Shape of [[NDArray]] or other data
 */
class Shape(dims: Traversable[Long]) extends Serializable {
  private val shape = dims.toVector

  def this(dims: Long*) = {
    this(dims.toVector)
  }

  def apply(dim: Int): Long = shape(dim)
  def size: Int = shape.size
  def length: Int = shape.length
  def drop(dim: Int): Shape = new Shape(shape.drop(dim))
  def slice(from: Int, end: Int): Shape = new Shape(shape.slice(from, end))
  def product: Long = shape.product
  def head: Long = shape.head

  def ++(other: Shape): Shape = new Shape(shape ++ other.shape)

  def toArray: Array[Long] = shape.toArray
  def toVector: Vector[Long] = shape

  override def toString(): String = s"(${shape.mkString(",")})"

  override def equals(o: Any): Boolean = o match {
    case that: Shape =>
      that != null && that.shape.sameElements(shape)
    case _ => false
  }

  override def hashCode(): Int = {
    shape.hashCode()
  }
}

object Shape {
  def apply(dims: Long *): Shape = new Shape(dims: _*)
  def apply(dims: Traversable[Long]): Shape = new Shape(dims)
}

