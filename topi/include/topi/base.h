/*!
 *  Copyright (c) 2017 by Contributors
 * \brief Base header for TOPI library
 * \file topi/base.h
 */
#ifndef TOPI_BASE_H_
#define TOPI_BASE_H_

#ifndef TOPI_DLL
#ifdef _WIN32
#ifdef tvm_topi_EXPORTS
#define TOPI_DLL __declspec(dllexport)
#else
#define TOPI_DLL __declspec(dllimport)
#endif
#else
#define TOPI_DLL
#endif
#endif

namespace topi {

TOPI_DLL void ensure_lib_loaded();

}  // namespace topi
#endif  // TOPI_BASE_H_
