/*!
 *  Copyright (c) 2017 by Contributors
 * \file tvm/target_info.h
 * \brief Various information about target.
 */
#ifndef TVM_TARGET_INFO_H_
#define TVM_TARGET_INFO_H_

#include <string>
#include "base.h"
#include "expr.h"

namespace tvm {

/*!
 * \brief Memory information of special memory region.
 *  Use MemoryInfo as its container type
 */
struct MemoryInfoNode : public Node {
  /*! \brief The addressable unit */
  int unit_bits;
  /*! \brief Maximum number of bits supported in the memory */
  int max_num_bits;
  /*! \brief maximum number of bits to be used in simd op */
  int max_simd_bits;
  /*!
   * \brief head address of the buffer, if visible to CPU
   *  This address can be None.
   */
  Expr head_address;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("unit_bits", &unit_bits);
    v->Visit("max_num_bits", &max_num_bits);
    v->Visit("max_simd_bits", &max_simd_bits);
    v->Visit("head_address", &head_address);
  }

  static constexpr const char* _type_key = "MemoryInfo";
  TVM_DECLARE_NODE_TYPE_INFO(MemoryInfoNode, Node);
};

/*! \brief Defines memory info */
TVM_DEFINE_NODE_REF(MemoryInfo, MemoryInfoNode);

/*!
 * \brief get memory info given scope
 * \param scope The scope name.
 * \return info The memory info.
 */
TVM_DLL MemoryInfo GetMemoryInfo(const std::string& scope);

}  // namespace tvm
#endif  // TVM_TARGET_INFO_H_
